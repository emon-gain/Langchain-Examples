import {ChatOpenAI} from '@langchain/openai'
import {ChatPromptTemplate} from "@langchain/core/prompts";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import {createRetrievalChain} from "langchain/chains/retrieval";
import {CheerioWebBaseLoader} from "langchain/document_loaders/web/cheerio";
import {RecursiveCharacterTextSplitter} from "langchain/text_splitter";
import {OpenAIEmbeddings} from "@langchain/openai";
import {MemoryVectorStore} from "langchain/vectorstores/memory";
import {
  PGVectorStore,
} from "@langchain/community/vectorstores/pgvector";
import {z} from 'zod'
import * as dotenv from 'dotenv'
import {StructuredOutputParser} from "langchain/output_parsers";
import {DirectoryLoader} from "langchain/document_loaders/fs/directory";
import {PDFLoader} from "langchain/document_loaders/fs/pdf";
import {StringOutputParser} from "@langchain/core/output_parsers";
import * as fs from "fs";
import {S3Loader} from "langchain/document_loaders/web/s3";
dotenv.config()
import pkg from 'pg';
import {TypeORMVectorStore} from "@langchain/community/vectorstores/typeorm";
import {RefineDocumentsChain} from "langchain/chains";

const model = new ChatOpenAI({
  openAIApiKey: process.env.OPENAI_API_KEY,
  modelName: 'gpt-4-turbo',
  temperature: 0
})

const prompt = ChatPromptTemplate.fromTemplate(`
  Answer the user's question. Act as an hr and provide the most effective candidate for the job position. Response me as precise as possible.
  format_instructions: {format_instructions}
  Context: {context}
  Question: {input}
`)

const outputParser = StructuredOutputParser.fromNamesAndDescriptions({
  job_desc: "job description"
});

const zodParser = StructuredOutputParser.fromZodSchema(
  z.object({
    // job_desc: z.string().describe("job description"),
    candidate_info: z.array(z.object({
      name: z.string().describe("candidate name"),
      contactInfo: z.string().describe("contact information"),
      email: z.string().email().describe("email address"),
      skills: z.array(z.string()).describe("skills"),
      matchingSkills: z.array(z.string()).describe("matching skills by job description and candidate skills"),
      requiredSkills: z.array(z.string()).describe("required skills for the job"),
      experiences: z.string().describe("number of year of working experience as part of a job. Provide 0 if no experience. Please do not count university, college, school, or projects as experience. Do not count projects or personal work as experience. Also ignore certification as experience."),
      most_recent_job_position: z.string().describe("most recent job position from experience. Provide 'none' if no experience. Please do not count university, college, school, or projects as experience. Do not count projects or personal work as experience. Also ignore certification as experience."),
      experience_details: z.string().describe("Provide information about working experience as well. Such as where and how long he worked as what position"),
    })).describe("candidate information"),
  })
)

const chain = await createStuffDocumentsChain({
  llm: model,
  prompt: prompt,
  outputParser: zodParser
})

const refinedChain = new RefineDocumentsChain({
  llmChain: chain,

})

const loader = new DirectoryLoader("resumes", {
  ".pdf": (path) => new PDFLoader(path, {
    parsedItemSeparator: "",
  })
})


const docs = await loader.load()

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 100
})

const splitDocs = await splitter.splitDocuments(docs)

console.log(splitDocs[0])
const embeddings = new OpenAIEmbeddings()
const config = {
  postgresConnectionOptions: {
    type: "postgres",
    host: "gainhqdatabase.cvnunyepijbg.eu-west-1.rds.amazonaws.com",
    port: 5432,
    user: "postgres",
    password: "YvDIyjUhSGW25lEJ",
    database: "longchain",
    ssl: {
      rejectUnauthorized: false,
      ca: fs.readFileSync('eu-west-1-bundle.pem').toString()
    }
  },
  tableName: "testlangchain",
  columns: {
    idColumnName: "id",
    vectorColumnName: "vector",
    contentColumnName: "content",
    metadataColumnName: "metadata",
  },
  distanceStrategy: "cosine"
};
const vectorStore = await PGVectorStore.initialize(
  embeddings,
  config
);
await vectorStore.ensureTableInDatabase();
await vectorStore.addDocuments(splitDocs)

console.log('added documents')
const retriever = vectorStore.asRetriever()

const retrievalChain = await createRetrievalChain({
  combineDocsChain: chain,
  retriever
})


const response = await retrievalChain.invoke({
  format_instructions: zodParser.getFormatInstructions(),
  // input: 'I want to sort all of my 3 candidates with higher recommended to lower recommended. Where i require skill of Web designing tools such as tailwind, css, html also a bit of knowledge in backend with node js also based on experience. Give me the candidates information with their EXPERIENCE details. Education is not experience. I want their total number of year in working experience. Last working position and give me none if he/she never have any real life working experience. Alongside i want there personal information such as email, contact number and name. Also give me the matching skills of the candidate with the job description. Please do not count education as experience.'
  input: 'Candidates should have good knowledge on React application development, code quality, UI/UX and testing.\n' +
    'Job Nature:\n' +
    'Job Nature: Full Time Working Hour: 11:00 AM to 08:00 PM Working days: Sunday to Thursday (5 days a week)\n' +
    'Job Location:\n' +
    'Mirpur 12, Dhaka.\n' +
    'Job requirements\n' +
    'You must have working experience on React Js.\n' +
    'Must Have to build app using React JS and GraphQL API.\n' +
    'Must Have Solid knowledge on ES6.\n' +
    'Must Have Write well designed, readable and efficient code.\n' +
    'Not Must have Excellent English communication skills.\n' +
    'Good knowledge in Git is required.\n' +
    'Review pull requests of team members.\n' +
    'Ensure good quality on UI/UX using predesigned template.\n' +
    'Excellent troubleshooting skills.\n' +
    'Knowledge on frontend testing tools.\n' +
    'Punctuality is an unquestionable requirement.\n' +
    '\n' +
    'I want 2 recommendation of the candidate who has the most matching skills with the job description. Ignore personal projects.I want the candidate information with their EXPERIENCE details. I want their total number of year in working experience without university education and projects return 0 if there are no working experience. Last working position and give me none if he/she never have any real life working experience. Alongside i want there personal information such as email, contact number and name. Also give me the matching skills of the candidate with the job description. Please do not count university, college, school, projects, or certificate as experience. Please note that designation is not a part of experience. Do not count projects or personal work as experience i want specific answer here. Check the experience from the main body of the resume not in header. Provide me information about working experience as well. Such as where and how long he worked as what position'
  // input: 'I want 2 recommendation for web developer position and want information from all candidates documents. Working experience without projects or university, name, contact info, email, skills, matching skills, required skills, most recent job position, experience details'
})

console.log(JSON.stringify(response.answer))

// console.log(response.answer)

fs.writeFileSync('response.json', JSON.stringify(response.answer), 'utf-8')

