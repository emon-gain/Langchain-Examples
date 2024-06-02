import * as dotenv from "dotenv";
dotenv.config();

import readline from "readline";

import { ChatOpenAI } from "@langchain/openai";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";

import { HumanMessage, AIMessage } from "@langchain/core/messages";

import { createOpenAIFunctionsAgent, AgentExecutor } from "langchain/agents";

// Tool imports
import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import { createRetrieverTool } from "langchain/tools/retriever";

// Custom Data Source, Vector Stores
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OpenAIEmbeddings } from "@langchain/openai";
import {DirectoryLoader} from "langchain/document_loaders/fs/directory";
import {PDFLoader} from "langchain/document_loaders/fs/pdf";
import fs from "fs";
import {PGVectorStore} from "@langchain/community/vectorstores/pgvector";
import {WebBrowser} from "langchain/tools/webbrowser";
import {PostgresChatMessageHistory} from "@langchain/community/stores/message/postgres";

// Create Retriever
const loader = new DirectoryLoader("resumes", {
  ".pdf": (path) => new PDFLoader(path, {
    parsedItemSeparator: "",
  })
})
const docs = await loader.load();

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 500,
  chunkOverlap: 20
})

const splitDocs = await splitter.splitDocuments(docs);

const embeddings = new OpenAIEmbeddings();

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
    sessionColumnName: "session",
    metadataColumnName: "metadata",
  },
  distanceStrategy: "cosine"
};

const vectorStore = await PGVectorStore.initialize(
  embeddings,
  config
);

await vectorStore.ensureTableInDatabase();
// await vectorStore.addDocuments(splitDocs)
const retriever = vectorStore.asRetriever()
// Instantiate the model
const model = new ChatOpenAI({
  modelName: "gpt-4-turbo",
  temperature: 0.2,
});

// Prompt Template
const prompt = ChatPromptTemplate.fromMessages([
  ("system", "You are a helpful assistant."),
  new MessagesPlaceholder("chat_history"),
  ("human", "{input}"),
  new MessagesPlaceholder("agent_scratchpad"),
]);

// Tools
const searchTool = new TavilySearchResults();
const retrieverTool = createRetrieverTool(retriever, {
  name: "lcel_search",
  description:
    "Use this tool for information about documents or resume or cv or candidates or person",
});
const webBrowserTool = new WebBrowser({
  model: model,
  embeddings: embeddings,
})

const tools = [searchTool, retrieverTool, webBrowserTool];

const agent = await createOpenAIFunctionsAgent({
  llm: model,
  prompt,
  tools,
});

// Create the executor
const agentExecutor = new AgentExecutor({
  agent,
  tools,
});

// User Input

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

const chatHistory = new PostgresChatMessageHistory({
  tableName: "langchain_chat_histories",
  sessionId: "lc-example",
  pool: vectorStore.pool
});

const chat_history = await chatHistory.getMessages()
const chatContentList = chat_history.map((chat) => {
  if (chat instanceof HumanMessage) {
    return {
      type: "human",
      content: chat.content,
    };
  } else if (chat instanceof AIMessage) {
    return {
      type: "ai",
      content: chat.content,
    };

  }
});
console.log(chatContentList)
function askQuestion() {
  rl.question("User: ", async (input) => {
    if (input.toLowerCase() === "exit") {
      rl.close();
      return;
    }

    const response = await agentExecutor.invoke({
      input: input,
      chat_history: chat_history,
    });

    console.log("Agent: ", response.output);

    await chatHistory.addUserMessage(input);
    await chatHistory.addAIMessage(response.output)

    askQuestion();
  });
}

askQuestion();
