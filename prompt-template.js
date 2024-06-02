import {ChatOpenAI} from '@langchain/openai'
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StructuredOutputParser } from "langchain/output_parsers";
import * as dotenv from 'dotenv'

dotenv.config()

const model = new ChatOpenAI({
  openAIApiKey: process.env.OPENAI_API_KEY,
  modelName: 'gpt-3.5-turbo',
  temperature: 0.7
})

const prompt = ChatPromptTemplate.fromMessages([
  ['system', 'Generate a joke based on a word provided by the user.'],
  ['human', '{input}']
])

//Create chain

const chain = prompt.pipe(model)

const response = await chain.invoke({
  input: 'dogs'
})

console.log(response)
