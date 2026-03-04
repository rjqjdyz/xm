import bcrypt
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, ForeignKey, Index, Text
from sqlalchemy.dialects.mysql import MEDIUMTEXT
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime

Base = declarative_base()


# === User 模型 ===
class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), nullable=False, unique=True)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.now())


class Document(Base):
    __tablename__ = 'documents'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)  # 关联用户
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    content = Column(MEDIUMTEXT, nullable=False)
    chunk_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.now())
    is_active = Column(Boolean, default=True)

    parent_chunks = relationship("ParentChunk", back_populates="document", cascade="all, delete-orphan")
    child_chunks = relationship("ChildChunk", back_populates="document", cascade="all, delete-orphan")


class ParentChunk(Base):
    __tablename__ = 'parent_chunks'

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey('documents.id'), nullable=False)
    parent_id = Column(String(100), nullable=False, unique=True)
    content = Column(Text, nullable=False)
    json_metadata = Column(Text)
    vector_id = Column(String(100))
    created_at = Column(DateTime, default=datetime.now())

    document = relationship("Document", back_populates="parent_chunks")
    child_chunks = relationship("ChildChunk", back_populates="parent_chunk", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_parent_document_id', 'document_id'),
        Index('idx_parent_id', 'parent_id'),
    )


class ChildChunk(Base):
    __tablename__ = 'child_chunks'

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey('documents.id'), nullable=False)
    parent_chunk_id = Column(Integer, ForeignKey('parent_chunks.id'), nullable=False)
    child_id = Column(String(100), nullable=False)
    content = Column(Text, nullable=False)
    json_metadata = Column(Text)
    vector_id = Column(String(100))
    created_at = Column(DateTime, default=datetime.now())

    document = relationship("Document", back_populates="child_chunks")
    parent_chunk = relationship("ParentChunk", back_populates="child_chunks")

    __table_args__ = (
        Index('idx_child_document_id', 'document_id'),
        Index('idx_child_parent_id', 'parent_chunk_id'),
        Index('idx_child_id', 'child_id'),
    )


class ChatHistory(Base):
    __tablename__ = 'chat_history'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)  # 关联用户
    session_id = Column(String(100), nullable=False)
    user_message = Column(Text, nullable=False)
    assistant_message = Column(Text, nullable=False)
    document_ids = Column(String(500))
    used_chunks = Column(Text)
    created_at = Column(DateTime, default=datetime.now())

    __table_args__ = (
        Index('idx_chat_session_id', 'session_id'),
        Index('idx_chat_created_at', 'created_at'),
    )


class DatabaseManager:
    def __init__(self, config):
        self.config = config
        self.engine = None
        self.SessionLocal = None
        self.init_database()

    def init_database(self):
        connection_string = f"mysql+pymysql://{self.config.MYSQL_USER}:{self.config.MYSQL_PASSWORD}@{self.config.MYSQL_HOST}:{self.config.MYSQL_PORT}/{self.config.MYSQL_DATABASE}?charset=utf8mb4"

        try:
            self.engine = create_engine(connection_string, echo=False)
            Base.metadata.create_all(self.engine)
            self.SessionLocal = sessionmaker(bind=self.engine)
            print("数据库连接成功")
        except Exception as e:
            print(f"数据库连接失败: {e}")
            raise

    def get_session(self):
        return self.SessionLocal()

    # === 用户注册 ===
    def register_user(self, username, password):
        session = self.get_session()
        try:
            existing_user = session.query(User).filter(User.username == username).first()
            if existing_user:
                return False, "用户名已存在"

            salt = bcrypt.gensalt()
            hashed_pw = bcrypt.hashpw(password.encode('utf-8'), salt)

            new_user = User(username=username, password_hash=hashed_pw.decode('utf-8'))
            session.add(new_user)
            session.commit()
            return True, "注册成功"
        except Exception as e:
            session.rollback()
            return False, str(e)
        finally:
            session.close()

    # === 用户登录 ===
    def login_user(self, username, password):
        session = self.get_session()
        try:
            user = session.query(User).filter(User.username == username).first()
            if user:
                if bcrypt.checkpw(password.encode('utf-8'), user.password_hash.encode('utf-8')):
                    return True, user
            return False, None
        finally:
            session.close()

    # === 保存文档 ===
    def save_document_with_chunks(self, filename, file_path, content, parent_docs, child_docs, parent_vector_ids,
                                  child_vector_ids, user_id):
        session = self.get_session()
        try:
            doc = Document(
                user_id=user_id,
                filename=filename,
                file_path=file_path,
                content=content,
                chunk_count=len(child_docs)
            )
            session.add(doc)
            session.flush()
            doc_id = doc.id

            parent_chunk_map = {}
            for i, (parent_doc, vector_id) in enumerate(zip(parent_docs, parent_vector_ids)):
                parent_chunk = ParentChunk(
                    document_id=doc_id,
                    parent_id=parent_doc.metadata.get('parent_id', f'parent_{i}'),
                    content=parent_doc.page_content,
                    json_metadata=str(parent_doc.metadata),
                    vector_id=vector_id
                )
                session.add(parent_chunk)
                session.flush()
                parent_chunk_map[parent_chunk.parent_id] = parent_chunk.id

            for child_doc, vector_id in zip(child_docs, child_vector_ids):
                parent_id = child_doc.metadata.get('parent_id', 'unknown')
                parent_chunk_id = parent_chunk_map.get(parent_id)

                child_chunk = ChildChunk(
                    document_id=doc_id,
                    parent_chunk_id=parent_chunk_id,
                    child_id=child_doc.metadata.get('child_id', f'child_{len(child_docs)}'),
                    content=child_doc.page_content,
                    json_metadata=str(child_doc.metadata),
                    vector_id=vector_id
                )
                session.add(child_chunk)

            session.commit()
            return doc_id

        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    # === 获取文档列表 ===
    def get_all_documents(self, user_id):
        session = self.get_session()
        try:
            docs = session.query(Document).filter(
                Document.is_active == True,
                Document.user_id == user_id
            ).all()
            return docs
        finally:
            session.close()

    # === 获取特定会话的历史 ===
    def get_chat_history(self, session_id, limit=10):
        session = self.get_session()
        try:
            chats = session.query(ChatHistory).filter(
                ChatHistory.session_id == session_id
            ).order_by(ChatHistory.created_at.desc()).limit(limit).all()
            return list(reversed(chats))
        finally:
            session.close()

    # === 获取用户的所有会话列表 ===
    def get_user_sessions(self, user_id):
        """获取用户的所有会话列表（按时间倒序）"""
        session = self.get_session()
        try:
            chats = session.query(ChatHistory).filter(
                ChatHistory.user_id == user_id
            ).order_by(ChatHistory.created_at.desc()).all()

            sessions = []
            seen_ids = set()
            for chat in chats:
                if chat.session_id not in seen_ids:
                    sessions.append(chat)
                    seen_ids.add(chat.session_id)
            return sessions
        finally:
            session.close()

    # === 保存对话记录 ===
    def save_chat_history(self, session_id, user_message, assistant_message, user_id, document_ids=None,
                          used_chunks=None):
        session = self.get_session()
        try:
            chat = ChatHistory(
                session_id=session_id,
                user_id=user_id,
                user_message=user_message,
                assistant_message=assistant_message,
                document_ids=document_ids,
                used_chunks=used_chunks
            )
            session.add(chat)
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    # === 物理删除文档 ===
    def delete_document(self, doc_id):
        session = self.get_session()
        try:
            doc = session.query(Document).filter(Document.id == doc_id).first()
            if doc:
                session.delete(doc)
                session.commit()
                return True
            return False
        except Exception as e:
            session.rollback()
            print(f"数据库删除失败: {e}")
            raise e
        finally:
            session.close()

    # === 新增：删除指定会话 ===
    def delete_session(self, session_id):
        """物理删除指定会话的所有记录"""
        session = self.get_session()
        try:
            # 批量删除该 session_id 的所有记录
            session.query(ChatHistory).filter(ChatHistory.session_id == session_id).delete()
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            print(f"删除会话失败: {e}")
            return False
        finally:
            session.close()