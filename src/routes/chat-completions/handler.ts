import type { Context } from "hono"

import { streamSSE, type SSEMessage } from "hono/streaming"

import { awaitApproval } from "~/lib/approval"
import { createHandlerLogger } from "~/lib/logger"
import { checkRateLimit } from "~/lib/rate-limit"
import { state } from "~/lib/state"
import { getTokenCount } from "~/lib/tokenizer"
import { isNullish } from "~/lib/utils"
import {
  translateOpenAIPayloadToResponsesPayload,
  translateResponsesResultToOpenAIResponse,
  translateResponsesStreamEventToOpenAIChunk,
} from "~/routes/chat-completions/responses-translation"
import { getResponsesRequestOptions } from "~/routes/responses/utils"
import {
  createChatCompletions,
  type ChatCompletionResponse,
  type ChatCompletionsPayload,
} from "~/services/copilot/create-chat-completions"
import {
  createResponses,
  type ResponsesResult,
  type ResponseStreamEvent,
} from "~/services/copilot/create-responses"

const logger = createHandlerLogger("chat-completions-handler")

export async function handleCompletion(c: Context) {
  await checkRateLimit(state)

  let payload = await c.req.json<ChatCompletionsPayload>()
  logger.debug("Request payload:", JSON.stringify(payload).slice(-400))

  // Find the selected model
  const selectedModel = state.models?.data.find(
    (model) => model.id === payload.model,
  )

  // Calculate and display token count
  try {
    if (selectedModel) {
      const tokenCount = await getTokenCount(payload, selectedModel)
      logger.info("Current token count:", tokenCount)
    } else {
      logger.warn("No model selected, skipping token count calculation")
    }
  } catch (error) {
    logger.warn("Failed to calculate token count:", error)
  }

  if (state.manualApprove) await awaitApproval()

  if (isNullish(payload.max_tokens)) {
    payload = {
      ...payload,
      max_tokens: selectedModel?.capabilities.limits.max_output_tokens,
    }
    logger.debug("Set max_tokens to:", JSON.stringify(payload.max_tokens))
  }

  const useResponsesApi = shouldUseResponsesApi(payload.model)

  if (useResponsesApi) {
    return handleWithResponsesApi(c, payload)
  }

  const response = await createChatCompletions(payload)

  if (isNonStreaming(response)) {
    if (response.choices.length > 0) {
      const message = response.choices[0].message
      if (message.reasoning_text) {
        // @ts-expect-error - reasoning_content is not in the type definition
        message.reasoning_content = message.reasoning_text
        delete message.reasoning_text
      }
    }
    logger.debug("Non-streaming response:", JSON.stringify(response))
    return c.json(response)
  }

  logger.debug("Streaming response")
  return streamSSE(c, async (stream) => {
    for await (const chunk of response) {
      processCopilotChunk(chunk)
      logger.debug("Streaming chunk:", JSON.stringify(chunk))
      await stream.writeSSE(chunk as SSEMessage)
    }
  })
}

const RESPONSES_ENDPOINT = "/responses"

const shouldUseResponsesApi = (modelId: string): boolean => {
  const selectedModel = state.models?.data.find((model) => model.id === modelId)
  return (
    selectedModel?.supported_endpoints?.includes(RESPONSES_ENDPOINT) ?? false
  )
}

const handleWithResponsesApi = async (
  c: Context,
  payload: ChatCompletionsPayload,
) => {
  const responsesPayload = translateOpenAIPayloadToResponsesPayload(payload)
  logger.debug(
    "Translated Responses payload:",
    JSON.stringify(responsesPayload).slice(-400),
  )

  const { vision, initiator } = getResponsesRequestOptions(responsesPayload)
  const response = await createResponses(responsesPayload, {
    vision,
    initiator,
  })

  if (payload.stream && isAsyncIterable(response)) {
    logger.debug("Streaming response from Copilot (Responses API)")
    return streamSSE(c, async (stream) => {
      for await (const chunk of response) {
        if (chunk.event === "ping" || !chunk.data) continue

        const chunkData = processResponsesApiChunk(chunk.data, payload.model)
        if (chunkData) {
          logger.debug("Translated OpenAI chunk:", chunkData)
          await stream.writeSSE({
            data: chunkData,
          })
        }
      }
      // Send [DONE]
      await stream.writeSSE({ data: "[DONE]" })
    })
  }

  // Non-streaming
  logger.debug(
    "Non-streaming Responses result:",
    JSON.stringify(response).slice(-400),
  )
  const openAIResponse = translateResponsesResultToOpenAIResponse(
    response as ResponsesResult,
  )

  if (openAIResponse.choices.length > 0) {
    const message = openAIResponse.choices[0].message
    if (message.reasoning_text) {
      // @ts-expect-error - reasoning_content
      message.reasoning_content = message.reasoning_text
      delete message.reasoning_text
    }
  }

  logger.debug("Translated OpenAI response:", JSON.stringify(openAIResponse))
  return c.json(openAIResponse)
}

const isNonStreaming = (
  response: Awaited<ReturnType<typeof createChatCompletions>>,
): response is ChatCompletionResponse => Object.hasOwn(response, "choices")

const isAsyncIterable = <T>(value: unknown): value is AsyncIterable<T> =>
  Boolean(value)
  && typeof (value as AsyncIterable<T>)[Symbol.asyncIterator] === "function"

function processCopilotChunk(chunk: { data?: string }) {
  if (!chunk.data || chunk.data === "[DONE]") return

  try {
    const data = JSON.parse(chunk.data) as {
      choices?: Array<{
        delta?: {
          reasoning_text?: string
          reasoning_content?: string | null
        }
      }>
    }
    if (data.choices?.[0]?.delta) {
      const delta = data.choices[0].delta
      if (delta.reasoning_text) {
        delta.reasoning_content = delta.reasoning_text
        delete delta.reasoning_text
      } else {
        delta.reasoning_content = null
      }
      chunk.data = JSON.stringify(data)
    }
  } catch (error) {
    logger.warn("Failed to parse chunk data:", error)
  }
}

function processResponsesApiChunk(data: string, model: string): string | null {
  try {
    const event = JSON.parse(data) as ResponseStreamEvent
    const openAIChunk = translateResponsesStreamEventToOpenAIChunk(event, model)

    if (openAIChunk) {
      if (openAIChunk.choices.length > 0) {
        const delta = openAIChunk.choices[0].delta
        if (delta.reasoning_text) {
          // @ts-expect-error - reasoning_content
          delta.reasoning_content = delta.reasoning_text
          delete delta.reasoning_text
        }
      }
      return JSON.stringify(openAIChunk)
    }
  } catch (error) {
    logger.warn("Failed to process stream chunk:", error)
  }
  return null
}
