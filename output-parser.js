import {ChatOpenAI} from '@langchain/openai'
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser, CommaSeparatedListOutputParser } from "@langchain/core/output_parsers";
import { StructuredOutputParser } from "langchain/output_parsers";
import {z} from 'zod'
import * as dotenv from 'dotenv'

dotenv.config()

const model = new ChatOpenAI({
  openAIApiKey: process.env.OPENAI_API_KEY,
  modelName: 'gpt-3.5-turbo',
  temperature: 0.7
})

const callStringOutputParser = async () => {
  const prompt = ChatPromptTemplate.fromMessages([
    ['system', 'Generate a joke based on a word provided by the user.'],
    ['human', '{input}']
  ])

  const parser = new StringOutputParser()


//Create chain

  const chain = prompt.pipe(model).pipe(parser)

  return  chain.invoke({
    input: 'dogs'
  })
}

const callListOutputParser = async () => {
  const prompt = ChatPromptTemplate.fromTemplate(`
    Provide 5 synonyms, seperated by commas, for the following word {input}".
  `)

  const parser = new CommaSeparatedListOutputParser()

  const chain = prompt.pipe(model).pipe(parser)

  return chain.invoke({
    input: 'happy'
  })

}

const  callStructuredParser = async () => {
  const prompt = ChatPromptTemplate.fromTemplate(
    "Extract information from the following phrase.\n{format_instructions}\n{phrase}"
  );

  const outputParser = StructuredOutputParser.fromNamesAndDescriptions({
    name: "name of the person",
    age: "age of person",
  });

  const chain = prompt.pipe(model).pipe(outputParser);

  return await chain.invoke({
    phrase: "Max is 30 years old",
    format_instructions: outputParser.getFormatInstructions(),
  });
}

const callZodOutputParser = async () => {
  const prompt = ChatPromptTemplate.fromTemplate(
    "Extract information from the following phrase.\n{format_instructions}\n{phrase}"
  );

  const outputParser = StructuredOutputParser.fromZodSchema(
    z.object({
      recipe: z.string().describe("Name of the recipe"),
      ingredients: z.array(z.string()).describe("List of ingredients"),
    })
  );

  const chain = prompt.pipe(model).pipe(outputParser);

  return await chain.invoke({
    phrase: "The ingerdients for a Spagetti Bolognes are: flour, sugar, eggs, milk",
    format_instructions: outputParser.getFormatInstructions(),
  });
}

const response = await callZodOutputParser()
console.log(response)
