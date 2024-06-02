import { BedrockRuntimeClient } from "@aws-sdk/client-bedrock-runtime";
import { BedrockEmbeddings } from "langchain/embeddings/bedrock";
import fs from "fs";
import {PGVectorStore} from "@langchain/community/vectorstores/pgvector";
import {DirectoryLoader} from "langchain/document_loaders/fs/directory";
import {PDFLoader} from "langchain/document_loaders/fs/pdf";
import {RecursiveCharacterTextSplitter} from "langchain/text_splitter";
import * as dotenv from 'dotenv'
dotenv.config()

const client = new BedrockRuntimeClient({
  region: "us-east-1",
  credentials: {
    accessKeyId: process.env.AWS_ACCESS_KEY,
    secretAccessKey: process.env.AWS_SECRET_KEY,
  },
});
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

const embeddings = new BedrockEmbeddings({
  client,
  model: "amazon.titan-embed-text-v1",
});
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
await vectorStore.end()
