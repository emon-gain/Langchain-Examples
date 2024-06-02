import {ChatOpenAI, OpenAIEmbeddings} from "@langchain/openai";
import pg from "pg";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import {createOpenAIFunctionsAgent, AgentExecutor} from "langchain/agents";
import {TavilySearchResults} from "@langchain/community/tools/tavily_search";
import { WebBrowser } from "langchain/tools/webbrowser";
import {createRetrieverTool} from "langchain/tools/retriever";

import {HumanMessage, AIMessage} from "@langchain/core/messages";

import {CheerioWebBaseLoader} from "langchain/document_loaders/web/cheerio";
import {RecursiveCharacterTextSplitter} from "langchain/text_splitter";
import {MemoryVectorStore} from "langchain/vectorstores/memory";
import { HNSWLib } from "langchain/vectorstores/hnswlib";

import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";

import readline from "readline";
// Import environment variables
import * as dotenv from "dotenv";
import {FaissStore} from "@langchain/community/vectorstores/faiss";
import fs from "fs";
import { v4 } from 'uuid'
import {PGVectorStore} from "@langchain/community/vectorstores/pgvector";
import {PostgresChatMessageHistory} from "@langchain/community/stores/message/postgres";
import {TypeORMVectorStore} from "@langchain/community/vectorstores/typeorm";
dotenv.config();

const loader = new DirectoryLoader("resumes", {
  ".pdf": (path) => new PDFLoader(path, {
    parsedItemSeparator: "",
  })
})

const docs = await loader.load()

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 500,
  chunkOverlap: 20
})

const splitDocs = await splitter.splitDocuments(docs)
console.log(splitDocs[0])
const embeddings = new OpenAIEmbeddings()
// const args = {
//   postgresConnectionOptions: {
//     type: "postgres",
//     host: "gainhqdatabase.cvnunyepijbg.eu-west-1.rds.amazonaws.com",
//     port: 5432,
//     user: "postgres",
//     password: "YvDIyjUhSGW25lEJ",
//     database: "longchain",
//     tableName: "testlangchain",
//     ssl: {
//       ca: fs.readFileSync('eu-west-1-bundle.pem').toString()
//     },
//     extra: {
//       trustServerCertificate: true,
//     }
//   }
// };
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
    sessionColumnName: "session"
  },
  distanceStrategy: "cosine"
};
const sessionId = v4()
splitDocs.map(doc => doc.metadata = {...doc.metadata, session: sessionId})
const vectorStore = await PGVectorStore.initialize(
  embeddings,
  config
);
await vectorStore.ensureTableInDatabase();
await vectorStore.addDocuments(splitDocs)
console.log('added documents')
// const result = await vectorStore.client.query('DELETE FROM testlangchain')
// console.log(result)
// // console.log(docs)
// const retriever = vectorStore.asRetriever()
// const model = new ChatOpenAI({
//   modelName: "gpt-4-turbo",
//   temperature: 0,
// });
//
// const prompt = ChatPromptTemplate.fromMessages([
//   ["system", "You are a helpful assistant called Max. You will act as an hr assistant and help me with the pdfs and scrap information from url"],
//   new MessagesPlaceholder("chat_history"),
//   ["human", "{input}"],
//   new MessagesPlaceholder("agent_scratchpad"),
// ]);
//
// // Define tools
// const retrieverTool = createRetrieverTool(retriever, {
//   name: "pdf_retriever",
//   description: 'Use this tools when searching information for documents or resume or cv or candidates or person',
// })
// const searchTool = new TavilySearchResults({
//   name: "tavily_search",
//   description: 'Use this tool to search for information on the web',
// })
//
// const webBrowserTool = new WebBrowser({
//   model: model,
//   embeddings: embeddings,
// })
// const tools = [retrieverTool, searchTool, webBrowserTool]
//
// //Create agent
// const agent = await createOpenAIFunctionsAgent({
//   llm: model,
//   prompt,
//   tools
// })
//
// const agentExecutor = new AgentExecutor({
//   agent,
//   tools
// })
//
// const chatHistory = new PostgresChatMessageHistory({
//   tableName: "langchain_chat_histories",
//   sessionId: "lc-example",
//   pool: vectorStore.pool
// });
//
// const chat_history = await chatHistory.getMessages()
// const query = "Can you suggest me 2 candidates for web designer based on the documents i provided?"
//
// await chatHistory.addUserMessage(query)
//
// const response = await agentExecutor.invoke({
//   input: query,
//   chat_history
// });
// await chatHistory.addAIMessage(response.output)
// console.log('Agent: ',response.output);
await vectorStore.end()
