from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import uuid


class RAGSystem:
    def __init__(self, config, db_manager, vector_store):
        self.config = config
        self.db_manager = db_manager
        self.vector_store = vector_store

        # 初始化LLM
        self.llm = ChatOpenAI(
            api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_BASE_URL,
            model="qwen-plus-2025-04-28",
            temperature=0.7,
            streaming=True,
        )

        # RAG提示模板
        self.rag_prompt = PromptTemplate(
            template="""你是一个智能助手。请基于以下上下文信息回答用户的问题。如果上下文中没有相关信息，请诚实地说明。

                上下文信息:
                {context}

                历史对话:
                {chat_history}

                用户问题: {question}

                请提供准确、有帮助的回答:""",
            input_variables=["context", "chat_history", "question"]
        )

    def generate_session_id(self):
        """生成会话ID"""
        return str(uuid.uuid4())

    def normal_chat(self, question, session_id, user_id):
        """普通对话（不使用文档）"""
        try:
            # 获取历史对话
            chat_history = self.db_manager.get_chat_history(session_id)

            # 构建对话上下文
            messages = []
            for chat in chat_history[-5:]:  # 最近5轮对话
                messages.append(f"Human: {chat.user_message}")
                messages.append(f"Assistant: {chat.assistant_message}")

            # 添加当前问题
            messages.append(f"Human: {question}")

            conversation_context = "\n".join(messages)

            # 生成回答
            response = self.llm.invoke(conversation_context + "\nAssistant:")
            answer = response.content

            # 保存对话历史（传入 user_id）
            self.db_manager.save_chat_history(
                session_id=session_id,
                user_message=question,
                assistant_message=answer,
                user_id=user_id
            )

            return answer

        except Exception as e:
            error_msg = f"普通对话出错: {str(e)}"
            print(error_msg)
            return error_msg


    def chat_with_documents(self, question, document_ids, session_id, user_id):
        """基于文档的RAG对话"""
        try:
            # 创建检索器
            retriever = self.vector_store.create_retriever(
                use_compression=True
            )
            # 检索相关文档  检索的是子文档
            retrieved_docs = retriever.invoke(question)
            # 获取父文档以提供更完整的上下文
            parent_docs = self.vector_store.get_parent_documents(retrieved_docs)

            # 构建上下文
            context = "\n\n".join(parent_docs)
            print("最终检索的上下文：", context)
            # 获取历史对话
            chat_history = self.db_manager.get_chat_history(session_id)
            history_text = "\n".join([
                f"用户: {chat.user_message}\n助手: {chat.assistant_message}"
                for chat in chat_history[-3:]  # 最近3轮对话
            ])

            # 构建完整提示
            full_prompt = self.rag_prompt.format(
                context=context,
                chat_history=history_text,
                question=question
            )

            # 生成回答
            response = self.llm.invoke(full_prompt)
            answer = response.content

            # 保存对话历史（传入 user_id）
            self.db_manager.save_chat_history(
                session_id=session_id,
                user_message=question,
                assistant_message=answer,
                document_ids=",".join(map(str, document_ids)),
                user_id=user_id
            )

            return answer, retrieved_docs

        except Exception as e:
            error_msg = f"RAG对话出错: {str(e)}"
            print(error_msg)
            return error_msg, []