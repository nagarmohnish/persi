"""Initial schema

Revision ID: 001
Revises:
Create Date: 2026-02-16

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Users
    op.create_table(
        "users",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("email", sa.String(), nullable=False, unique=True),
        sa.Column("full_name", sa.String(), nullable=True),
        sa.Column("avatar_url", sa.String(), nullable=True),
        sa.Column("onboarding_completed", sa.Boolean(), server_default="false"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )

    # Startups
    op.create_table(
        "startups",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("user_id", sa.String(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("name", sa.String(), nullable=True),
        sa.Column("one_liner", sa.String(), nullable=True),
        sa.Column("problem_statement", sa.Text(), nullable=True),
        sa.Column("target_audience", sa.String(), nullable=True),
        sa.Column("stage", sa.String(), server_default="idea"),
        sa.Column("industry", sa.String(), nullable=True),
        sa.Column("business_model", sa.String(), nullable=True),
        sa.Column("founding_team", postgresql.JSONB(), server_default="[]"),
        sa.Column("context_notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("idx_startups_user_id", "startups", ["user_id"])

    # Conversations
    op.create_table(
        "conversations",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("user_id", sa.String(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("startup_id", sa.String(), sa.ForeignKey("startups.id"), nullable=True),
        sa.Column("title", sa.String(), server_default="'New conversation'"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("idx_conversations_user_id", "conversations", ["user_id"])
    op.create_index("idx_conversations_updated_at", "conversations", ["updated_at"])

    # Messages
    op.create_table(
        "messages",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("conversation_id", sa.String(), sa.ForeignKey("conversations.id"), nullable=False),
        sa.Column("role", sa.String(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("tool_calls", postgresql.JSONB(), nullable=True),
        sa.Column("tool_results", postgresql.JSONB(), nullable=True),
        sa.Column("metadata", postgresql.JSONB(), server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("idx_messages_conversation_id", "messages", ["conversation_id"])
    op.create_index("idx_messages_created_at", "messages", ["created_at"])

    # Journey Tasks
    op.create_table(
        "journey_tasks",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("startup_id", sa.String(), sa.ForeignKey("startups.id"), nullable=False),
        sa.Column("stage", sa.String(), nullable=False),
        sa.Column("task_title", sa.String(), nullable=False),
        sa.Column("task_description", sa.Text(), nullable=True),
        sa.Column("status", sa.String(), server_default="'pending'"),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("ai_generated", sa.Boolean(), server_default="false"),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("idx_journey_tasks_startup_id", "journey_tasks", ["startup_id"])

    # Documents
    op.create_table(
        "documents",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("startup_id", sa.String(), sa.ForeignKey("startups.id"), nullable=False),
        sa.Column("doc_type", sa.String(), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("content", sa.Text(), nullable=True),
        sa.Column("file_url", sa.String(), nullable=True),
        sa.Column("version", sa.Integer(), server_default="1"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("idx_documents_startup_id", "documents", ["startup_id"])

    # Knowledge Chunks
    op.create_table(
        "knowledge_chunks",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("source", sa.String(), nullable=False),
        sa.Column("source_title", sa.String(), nullable=True),
        sa.Column("source_url", sa.String(), nullable=True),
        sa.Column("chunk_text", sa.Text(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), server_default="0"),
        sa.Column("stage_relevance", postgresql.ARRAY(sa.String()), server_default="{}"),
        sa.Column("topics", postgresql.ARRAY(sa.String()), server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("idx_knowledge_chunks_source", "knowledge_chunks", ["source"])

    # Seed dev user
    op.execute(
        "INSERT INTO users (id, email, full_name) VALUES ('dev-user-001', 'dev@persi.app', 'Dev User')"
    )


def downgrade() -> None:
    op.drop_table("knowledge_chunks")
    op.drop_table("documents")
    op.drop_table("journey_tasks")
    op.drop_table("messages")
    op.drop_table("conversations")
    op.drop_table("startups")
    op.drop_table("users")
