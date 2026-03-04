import time
import streamlit as st
from config.config import Config
from core.database import DatabaseManager
from core.document_processor import DocumentProcessor
from core.vector_store import VectorStore
from core.rag_system import RAGSystem
import os

# === 页面基础配置 ===
st.set_page_config(
    page_title="智能文档检索助手 - 毕业设计演示",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === 全局 CSS 样式优化 ===
st.markdown("""
<style>
    /* 全局字体与配色 */
    :root {
        --primary-color: #667eea;
        --secondary-color: #764ba2;
        --bg-color: #f8f9fa;
    }

    .stApp {
        background-color: var(--bg-color);
    }

    /* 登录页卡片样式 */
    .login-container {
        max-width: 450px;
        margin: 100px auto;
        padding: 40px;
        background: white;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.08);
        text-align: center;
    }

    .login-header {
        margin-bottom: 30px;
    }

    .login-header h1 {
        font-size: 28px;
        color: #333;
        margin-bottom: 10px;
    }

    .login-header p {
        color: #666;
        font-size: 14px;
    }

    /* 侧边栏样式优化 */
    section[data-testid="stSidebar"] {
        background-color: white;
        border-right: 1px solid #e1e5e9;
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    /* 知识库卡片 */
    .kb-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e1e5e9;
        margin-bottom: 15px;
        transition: all 0.2s;
    }

    .kb-card:hover {
        border-color: #667eea;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.1);
    }

    /* 隐藏默认元素 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def init_system():
    """初始化系统核心组件"""
    config = Config()
    db_manager = DatabaseManager(config)
    doc_processor = DocumentProcessor(config)
    vector_store = VectorStore(config)
    rag_system = RAGSystem(config, db_manager, vector_store)
    return config, db_manager, doc_processor, vector_store, rag_system


# === 核心逻辑函数 ===
def process_upload(uploaded_file, doc_processor, vector_store, db_manager, user_id):
    """处理文件上传"""
    with st.spinner(f"正在深入解析文档: {uploaded_file.name}..."):
        try:
            documents = doc_processor.load_document(uploaded_file)
            parent_docs, child_docs = doc_processor.create_parent_child_chunks(documents, uploaded_file.name)
            parent_vector_ids, child_vector_ids = vector_store.add_documents(parent_docs, child_docs, 0)

            content = "\n".join([doc.page_content for doc in documents])
            doc_id = db_manager.save_document_with_chunks(
                filename=uploaded_file.name,
                file_path="",
                content=content,
                parent_docs=parent_docs,
                child_docs=child_docs,
                parent_vector_ids=parent_vector_ids,
                child_vector_ids=child_vector_ids,
                user_id=user_id
            )
            return True, f"✅ 文档 '{uploaded_file.name}' 解析完成，已存入知识库。"
        except Exception as e:
            return False, f"❌ 处理失败: {str(e)}"


def stream_response(rag_system, message, selected_doc_ids, session_id, user_id):
    """生成流式响应 (修复中文逐字输出)"""
    if selected_doc_ids:
        response, retrieved_docs = rag_system.chat_with_documents(
            message, selected_doc_ids, session_id, user_id
        )
        # ✅ 修正：直接遍历字符串字符，实现中文流式效果
        for char in response:
            yield char, retrieved_docs
            time.sleep(0.01)  # 控制打字速度
    else:
        response = rag_system.normal_chat(message, session_id, user_id)
        # ✅ 修正：直接遍历字符串字符
        for char in response:
            yield char, None
            time.sleep(0.01)


# === 页面渲染函数 ===

def render_login_page(db_manager):
    """渲染登录/注册页"""
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("""
            <div class="login-container">
                <div class="login-header">
                    <h1>🎓 智能文档检索助手</h1>
                    <p>基于RAG和LangChain的问答交互系统</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["🔐 账号登录", "📝 用户注册"])

        with tab1:
            with st.form("login_form"):
                username = st.text_input("用户名", placeholder="请输入用户名")
                password = st.text_input("密码", type="password", placeholder="请输入密码")
                if st.form_submit_button("登 录", use_container_width=True):
                    success, user = db_manager.login_user(username, password)
                    if success:
                        st.session_state.logged_in = True
                        st.session_state.user_info = {"id": user.id, "username": user.username}
                        st.session_state.session_id = None
                        st.session_state.messages = []
                        st.rerun()
                    else:
                        st.error("用户名或密码错误")

        with tab2:
            with st.form("register_form"):
                new_user = st.text_input("设置用户名")
                new_pass = st.text_input("设置密码", type="password")
                confirm_pass = st.text_input("确认密码", type="password")
                if st.form_submit_button("注 册", use_container_width=True):
                    if new_pass != confirm_pass:
                        st.error("两次密码不一致")
                    else:
                        success, msg = db_manager.register_user(new_user, new_pass)
                        if success:
                            st.success("注册成功！请切换到登录页。")
                        else:
                            st.error(msg)


def render_sidebar_nav(username):
    """渲染侧边栏导航"""
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align: center; padding: 20px 0;">
            <div style="width: 60px; height: 60px; background: linear-gradient(135deg, #667eea, #764ba2); border-radius: 50%; margin: 0 auto; display: flex; align-items: center; justify-content: center; color: white; font-size: 24px;">
                {username[0].upper()}
            </div>
            <h3 style="margin-top: 10px; color: #333;">{username}</h3>
            <p style="color: #666; font-size: 12px;">在线 | 用户</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # 导航菜单
        selected_page = st.radio(
            "功能导航",
            ["🤖 智能对话", "📚 知识库管理", "📊 数据看板", "🕓 历史会话"],
            label_visibility="collapsed"
        )

        st.markdown("---")
        if st.button("🚪 退出登录", use_container_width=True):
            st.session_state.logged_in = False
            st.rerun()

        return selected_page


def render_chat_page(rag_system, db_manager, user_id):
    """渲染对话主界面"""
    documents = db_manager.get_all_documents(user_id)
    selected_doc_ids = []

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # === 侧边栏配置 ===
    with st.sidebar:
        if documents:
            with st.expander("📚 选择参考文档", expanded=True):
                doc_options = {d.filename: d.id for d in documents}
                selected_docs = st.multiselect(
                    "选择生效的文档:",
                    options=list(doc_options.keys()),
                    default=list(doc_options.keys())[:1] if documents else None,
                    label_visibility="collapsed",
                    key="chat_doc_selector"
                )
                selected_doc_ids = [doc_options[d] for d in selected_docs]

        st.markdown("<br>", unsafe_allow_html=True)
        sessions = db_manager.get_user_sessions(user_id)
        if sessions:
            with st.expander("🕓 切换历史会话", expanded=False):
                session_map = {f"{s.created_at.strftime('%m-%d %H:%M')} | {s.user_message[:8]}": s.session_id for s in
                               sessions}
                selected_sess_label = st.selectbox("选择会话", list(session_map.keys()), index=0, key="hist_selector")
                if st.button("加载", use_container_width=True):
                    st.session_state.session_id = session_map[selected_sess_label]
                    history = db_manager.get_chat_history(st.session_state.session_id)
                    st.session_state.messages = []
                    for h in history:
                        st.session_state.messages.append({"role": "user", "content": h.user_message})
                        st.session_state.messages.append({"role": "assistant", "content": h.assistant_message})
                    st.rerun()

        if st.button("➕ 新对话", use_container_width=True):
            st.session_state.session_id = rag_system.generate_session_id()
            st.session_state.messages = []
            st.rerun()

    # === 主内容区 ===
    st.markdown("### 🤖 智能文档问答")
    if selected_doc_ids:
        st.caption(f"已启用 RAG 模式，关联 {len(selected_doc_ids)} 个文档")
    else:
        st.caption("普通对话模式")

    # 1. 显示历史消息
    for msg in st.session_state.messages:
        role = msg["role"]
        with st.chat_message(role, avatar="👤" if role == "user" else "🤖"):
            st.write(msg["content"])
            if role == "assistant" and "docs" in msg and msg["docs"]:
                with st.expander("📚 参考来源"):
                    for d in msg["docs"]:
                        st.info(d.page_content[:150] + "...")

    # 2. 输入框与生成逻辑
    if prompt := st.chat_input("请输入问题..."):
        # 显示用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.write(prompt)

        # 生成回答
        with st.chat_message("assistant", avatar="🤖"):
            placeholder = st.empty()
            full_response = ""
            retrieved_docs = None

            # 确保 session_id 存在 (Bug修复)
            if st.session_state.session_id is None:
                st.session_state.session_id = rag_system.generate_session_id()

            # 调用流式生成
            generator = stream_response(rag_system, prompt, selected_doc_ids, st.session_state.session_id, user_id)

            for chunk, docs in generator:
                full_response += chunk
                placeholder.markdown(full_response + "▌")
                if docs:
                    retrieved_docs = docs

            placeholder.markdown(full_response)

            # 展示来源
            if retrieved_docs:
                with st.expander("📚 参考来源"):
                    for d in retrieved_docs:
                        st.info(d.page_content[:150] + "...")

            # 记录到历史
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "docs": retrieved_docs
            })


def render_kb_page(db_manager, doc_processor, vector_store, user_id):
    """渲染知识库管理页"""
    st.title("📚 知识库管理")
    st.caption("在此上传、管理您的文档资料，系统将自动进行向量化处理。")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 📤 文档上传")
        with st.container(border=True):
            uploaded_files = st.file_uploader(
                "拖拽或点击上传",
                type=['pdf', 'docx', 'txt'],
                accept_multiple_files=True
            )
            if uploaded_files:
                if st.button("开始处理", type="primary", use_container_width=True):
                    progress_bar = st.progress(0)
                    for i, file in enumerate(uploaded_files):
                        success, msg = process_upload(file, doc_processor, vector_store, db_manager, user_id)
                        if success:
                            st.toast(msg, icon="✅")
                        else:
                            st.error(msg)
                        progress_bar.progress((i + 1) / len(uploaded_files))
                    time.sleep(1)
                    st.rerun()

    with col2:
        st.markdown("### 📋 文档列表")
        documents = db_manager.get_all_documents(user_id)

        if not documents:
            st.info("暂无文档，请在左侧上传。")
        else:
            for doc in documents:
                with st.container(border=True):
                    c1, c2, c3, c4 = st.columns([3, 2, 1, 1])
                    c1.markdown(f"**📄 {doc.filename}**")
                    c2.text(f"📅 {doc.created_at.strftime('%Y-%m-%d')}")
                    c3.caption(f"{doc.chunk_count} 碎片")
                    if c4.button("🗑️", key=f"del_{doc.id}", help="删除此文档"):
                        vector_store.delete_document(doc.id)
                        db_manager.delete_document(doc.id)
                        st.success("删除成功")
                        time.sleep(0.5)
                        st.rerun()


def render_dashboard_page(db_manager, user_id):
    """渲染数据看板页"""
    st.title("📊 数据统计看板")
    st.markdown("---")

    docs = db_manager.get_all_documents(user_id)
    sessions = db_manager.get_user_sessions(user_id)

    # 核心指标
    c1, c2, c3 = st.columns(3)
    c1.metric("📚 知识库文档", f"{len(docs)} 篇")
    c2.metric("💬 历史对话", f"{len(sessions)} 次")
    total_chunks = sum([d.chunk_count for d in docs]) if docs else 0
    c3.metric("🧩 向量索引量", f"{total_chunks} 个")

    st.markdown("### 📈 数据洞察")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("文件类型分布")
        if docs:
            file_types = {}
            for d in docs:
                ext = d.filename.split('.')[-1].upper() if '.' in d.filename else '未知'
                file_types[ext] = file_types.get(ext, 0) + 1
            st.bar_chart(file_types, color="#667eea")
        else:
            st.info("暂无数据")

    with col2:
        st.subheader("最近活动")
        if sessions:
            st.dataframe(
                [{"时间": s.created_at.strftime('%Y-%m-%d %H:%M'), "摘要": s.user_message[:20]} for s in sessions[:5]],
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("暂无活动")


def render_history_page(db_manager, user_id):
    """渲染历史会话管理页"""
    st.title("🕓 历史会话管理")

    sessions = db_manager.get_user_sessions(user_id)
    if not sessions:
        st.info("暂无历史会话记录。")
        return

    # 批量删除功能
    with st.expander("🗑️ 批量删除模式", expanded=False):
        session_opts = {f"{s.created_at.strftime('%m-%d %H:%M')} | {s.user_message[:15]}...": s.session_id for s in
                        sessions}
        selected = st.multiselect("选择要删除的会话", list(session_opts.keys()))
        if st.button("确认删除选中项", type="primary"):
            count = 0
            for label in selected:
                sid = session_opts[label]
                if db_manager.delete_session(sid):
                    count += 1
            st.success(f"已删除 {count} 条记录")
            time.sleep(1)
            st.rerun()

    st.markdown("### 会话列表")
    for sess in sessions:
        with st.container(border=True):
            c1, c2 = st.columns([4, 1])
            with c1:
                st.markdown(f"**📅 {sess.created_at.strftime('%Y-%m-%d %H:%M')}**")
                st.caption(f"摘要: {sess.user_message[:50]}...")
            with c2:
                if st.button("👀 查看", key=f"view_{sess.session_id}"):
                    # 切换到对话页并加载
                    st.session_state.session_id = sess.session_id
                    history = db_manager.get_chat_history(sess.session_id)
                    st.session_state.messages = []
                    for h in history:
                        st.session_state.messages.append({"role": "user", "content": h.user_message})
                        st.session_state.messages.append({"role": "assistant", "content": h.assistant_message})
                    st.toast("会话已加载，请切换到【智能对话】页面查看", icon="✅")


# === 主程序入口 ===
def main():
    # 初始化
    config, db_manager, doc_processor, vector_store, rag_system = init_system()

    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    # 修复逻辑：不仅要检查键是否存在，还要检查值是否为 None
    if 'session_id' not in st.session_state or st.session_state.session_id is None:
        st.session_state.session_id = rag_system.generate_session_id()

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # 路由控制
    if not st.session_state.logged_in:
        render_login_page(db_manager)
    else:
        # 获取用户信息
        user_info = st.session_state.user_info

        # 渲染侧边栏并获取当前页面选择
        page = render_sidebar_nav(user_info['username'])

        # 页面分发
        if "智能对话" in page:
            # 所有的输入和渲染逻辑都在这个函数里，外部不再处理输入
            render_chat_page(rag_system, db_manager, user_info['id'])

        elif "知识库" in page:
            render_kb_page(db_manager, doc_processor, vector_store, user_info['id'])
        elif "看板" in page:
            render_dashboard_page(db_manager, user_info['id'])
        elif "历史" in page:
            render_history_page(db_manager, user_info['id'])


if __name__ == '__main__':
    main()