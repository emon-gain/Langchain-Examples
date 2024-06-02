import {ChatOpenAI, OpenAIEmbeddings} from "@langchain/openai";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";

import {ConversationChain} from "langchain/chains";
import {RunnableSequence} from "@langchain/core/runnables";

import {BufferMemory} from "langchain/memory";
import {UpstashRedisChatMessageHistory} from '@langchain/community/stores/message/upstash_redis'

import * as dotenv from "dotenv";
import readline from "readline";
import {AIMessage, HumanMessage} from "@langchain/core/messages";
dotenv.config();

const model = new ChatOpenAI({
  openAIApiKey: process.env.OPENAI_API_KEY,
  modelName: "gpt-3.5-turbo",
  temperature: 0.7,
});

const prompt = ChatPromptTemplate.fromTemplate(`
  Your are an Ai Assistant 
  History: {history}
  {input}
`);

const uptashChatHistory = new UpstashRedisChatMessageHistory({
  sessionId: 'chat1',
  config: {
    url: process.env.REDIS_URL,
    token: process.env.REDIS_TOKEN
  }
})

const memory = new BufferMemory({
  memoryKey: 'history',
  chatHistory: uptashChatHistory
})

// const chain = new ConversationChain({
//   llm: model,
//   prompt,
//   memory
// })

// const chain = prompt.pipe(model);
const chain = RunnableSequence.from([
  {
    input: (initialInput) => initialInput.input,
    memory: () => memory.loadMemoryVariables()
  },
  {
    input: (previousOutput) => previousOutput.input,
    history: (previousOutput) => previousOutput.memory.history,
  },
  prompt,
  model
])

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

const askQuestion = () => {

  rl.question("User: ", async (input) => {
    if (input === "exit") {
      rl.close();
    }
    else {
      const response = await chain.invoke({input});

      console.log('Agent: ',response.content);
      await memory.saveContext({input}, {
        output: response.content
      })
      askQuestion()
    }
  })
}

askQuestion()
