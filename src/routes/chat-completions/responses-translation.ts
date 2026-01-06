import { randomUUID } from "node:crypto"

import { getReasoningEffortForModel } from "~/lib/config"
import {
  type ChatCompletionChunk,
  type ChatCompletionResponse,
  type ChatCompletionsPayload,
  type Message,
  type Tool as OpenAITool,
} from "~/services/copilot/create-chat-completions"
import {
  type ResponseInputItem,
  type ResponsesPayload,
  type ResponsesResult,
  type ResponseStreamEvent,
  type Tool,
  type ToolChoiceFunction,
  type ToolChoiceOptions,
} from "~/services/copilot/create-responses"

export const translateOpenAIPayloadToResponsesPayload = (
  payload: ChatCompletionsPayload,
): ResponsesPayload => {
  const input: Array<ResponseInputItem> = []

  for (const message of payload.messages) {
    if (message.role === "system") {
      // System messages are handled in 'instructions' usually, but can also be in input?
      // Anthropic translation puts system in 'instructions'.
      // If we have multiple system messages or interleaved, it might be tricky.
      // For now, let's collect system messages for 'instructions' and others for 'input'.
      // But iterate first.
    } else {
      input.push(...translateMessage(message))
    }
  }

  const systemMessage = payload.messages.find(
    (m) => m.role === "system",
  )?.content
  const instructions = typeof systemMessage === "string" ? systemMessage : null

  return {
    model: payload.model,
    input,
    instructions,
    tools: translateTools(payload.tools),
    tool_choice: translateToolChoice(payload.tool_choice),
    temperature: payload.temperature,
    top_p: payload.top_p,
    max_output_tokens: payload.max_tokens,
    stream: payload.stream,
    parallel_tool_calls: true, // Default to true for OpenAI?
    reasoning: {
      effort: getReasoningEffortForModel(payload.model),
      summary: "detailed", // Default to detailed for thinking
    },
    include: ["reasoning.encrypted_content"],
  }
}

const translateMessage = (message: Message): Array<ResponseInputItem> => {
  if (message.role === "tool") {
    // OpenAI 'tool' role -> Responses 'function_call_output'
    return [
      {
        type: "function_call_output",
        call_id: message.tool_call_id ?? "",
        output:
          typeof message.content === "string" ?
            message.content
          : JSON.stringify(message.content),
      },
    ]
  }

  if (message.role === "assistant" && message.tool_calls) {
    const items: Array<ResponseInputItem> = []
    if (message.content) {
      items.push({
        type: "message",
        role: "assistant",
        content:
          typeof message.content === "string" ?
            message.content
          : JSON.stringify(message.content),
      })
    }
    for (const toolCall of message.tool_calls) {
      items.push({
        type: "function_call",
        call_id: toolCall.id,
        name: toolCall.function.name,
        arguments: toolCall.function.arguments,
        status: "completed",
      } as ResponseInputItem)
    }
    return items
  }

  // User or simple assistant message
  const content = message.content

  return [
    {
      type: "message",
      role: message.role as "user" | "assistant" | "developer",
      content: content as string,
    },
  ]
}

const translateTools = (
  tools: Array<OpenAITool> | null | undefined,
): Array<Tool> | null => {
  if (!tools) return null
  return tools.map((t) => ({
    type: "function",
    name: t.function.name,
    description: t.function.description,
    parameters: t.function.parameters,
    strict: false,
  }))
}

const translateToolChoice = (
  choice: unknown,
): ToolChoiceOptions | ToolChoiceFunction => {
  if (!choice || choice === "auto") return "auto"
  if (choice === "none") return "none"
  if (choice === "required") return "required"
  interface OpenAIToolChoice {
    type: "function"
    function: { name: string }
  }
  const c = choice as Partial<OpenAIToolChoice>
  if (c.type === "function" && c.function) {
    return {
      type: "function",
      name: c.function.name,
    }
  }
  return "auto"
}

// Streaming Translation Helpers
const createBaseChunk = (model: string): ChatCompletionChunk => ({
  id: "chatcmpl-" + randomUUID(),
  object: "chat.completion.chunk",
  created: Date.now(),
  model: model,
  choices: [],
})

const createOutputTextChunk = (
  base: ChatCompletionChunk,
  content: string,
): ChatCompletionChunk => ({
  ...base,
  choices: [
    {
      index: 0,
      delta: { content },
      finish_reason: null,
      logprobs: null,
    },
  ],
})

const createReasoningChunk = (
  base: ChatCompletionChunk,
  reasoning_text: string,
): ChatCompletionChunk => ({
  ...base,
  choices: [
    {
      index: 0,
      delta: {
        reasoning_text,
      },
      finish_reason: null,
      logprobs: null,
    },
  ],
})

const createToolCallChunk = (
  base: ChatCompletionChunk,
  call_id: string,
  name: string,
): ChatCompletionChunk => ({
  ...base,
  choices: [
    {
      index: 0,
      delta: {
        tool_calls: [
          {
            index: 0,
            id: call_id,
            type: "function",
            function: {
              name: name,
              arguments: "",
            },
          },
        ],
      },
      finish_reason: null,
      logprobs: null,
    },
  ],
})

const createFunctionArgumentsChunk = (
  base: ChatCompletionChunk,
  args: string,
): ChatCompletionChunk => ({
  ...base,
  choices: [
    {
      index: 0,
      delta: {
        tool_calls: [
          {
            index: 0,
            function: {
              arguments: args,
            },
          },
        ],
      },
      finish_reason: null,
      logprobs: null,
    },
  ],
})

const createCompletedChunk = (
  base: ChatCompletionChunk,
): ChatCompletionChunk => ({
  ...base,
  choices: [
    {
      index: 0,
      delta: {},
      finish_reason: "stop",
      logprobs: null,
    },
  ],
})

export const translateResponsesStreamEventToOpenAIChunk = (
  event: ResponseStreamEvent,
  model: string,
): ChatCompletionChunk | null => {
  const baseChunk = createBaseChunk(model)

  switch (event.type) {
    case "response.output_text.delta": {
      return createOutputTextChunk(baseChunk, event.delta)
    }

    case "response.reasoning_summary_text.delta": {
      return createReasoningChunk(baseChunk, event.delta)
    }

    case "response.output_item.added": {
      const item = event.item
      if (item.type === "function_call") {
        return createToolCallChunk(baseChunk, item.call_id, item.name)
      }
      return null
    }

    case "response.function_call_arguments.delta": {
      return createFunctionArgumentsChunk(baseChunk, event.delta)
    }

    case "response.completed": {
      return createCompletedChunk(baseChunk)
    }

    default: {
      return null
    }
  }
}

export const translateResponsesResultToOpenAIResponse = (
  result: ResponsesResult,
): ChatCompletionResponse => {
  // Simplistic mapping of output to choices
  let content = ""
  // We use unknown[] and cast items or define a minimal shape for tool_calls to avoid 'any'
  // Or we construct explicit objects.
  const tool_calls: Array<{
    id: string
    type: "function"
    function: { name: string; arguments: string }
  }> = []
  let reasoning_text = ""

  for (const item of result.output) {
    if (
      item.type === "message"  // Extract text from message content
      // item.content is Array<ResponseOutputContentBlock>
      && item.content
    ) {
      for (const block of item.content) {
        if (block.type === "output_text") {
          content += String(block.text)
        }
      }
    }
    if (item.type === "function_call") {
      tool_calls.push({
        id: item.call_id,
        type: "function",
        function: {
          name: item.name,
          arguments: item.arguments,
        },
      })
    }
    if (item.type === "reasoning" && item.summary) {
      for (const sum of item.summary) {
        if (sum.text) reasoning_text += sum.text
      }
    }
  }

  return {
    id: result.id,
    object: "chat.completion",
    created: result.created_at,
    model: result.model,
    choices: [
      {
        index: 0,
        message: {
          role: "assistant",
          content: content || null,
          // eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-assignment
          tool_calls: tool_calls.length > 0 ? (tool_calls as any) : undefined, // Cast to any to satisfy the complex ChatCompletionToolCall type if needed, or matched structure
          reasoning_text: reasoning_text || undefined,
        },
        finish_reason: tool_calls.length > 0 ? "tool_calls" : "stop",
        logprobs: null,
      },
    ],
  }
}
