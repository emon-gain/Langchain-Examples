import {ChatOpenAI} from '@langchain/openai'
import * as dotenv from 'dotenv'

dotenv.config()

const model = new ChatOpenAI({
  openAIApiKey: process.env.OPENAI_API_KEY,
  modelName: 'gpt-3.5-turbo',
  temperature: 0.7,
  maxTokens: 1000,
  verbose: true,
})

const response = await model.invoke('Write a poem about a rainy day.')
console.log(response)
