"""Phase 2A expansion — meetings, notes, calendar, travel, journey, contacts

Revision ID: 002
Revises: 001
Create Date: 2026-02-17

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ===== MEETINGS MODULE =====

    op.create_table(
        "meetings",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("user_id", sa.String(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("meeting_type", sa.String(), server_default="other"),
        sa.Column("status", sa.String(), server_default="scheduled"),
        sa.Column("start_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("end_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_minutes", sa.Integer(), nullable=True),
        sa.Column("location", sa.Text(), nullable=True),
        sa.Column("meeting_link", sa.Text(), nullable=True),
        sa.Column("platform", sa.String(), nullable=True),
        sa.Column("is_recurring", sa.Boolean(), server_default="false"),
        sa.Column("recurrence_rule", sa.Text(), nullable=True),
        sa.Column("startup_stage", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("idx_meetings_user_id", "meetings", ["user_id"])
    op.create_index("idx_meetings_start_time", "meetings", ["start_time"])
    op.create_index("idx_meetings_status", "meetings", ["status"])

    op.create_table(
        "meeting_participants",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("meeting_id", sa.String(), sa.ForeignKey("meetings.id"), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("email", sa.String(), nullable=True),
        sa.Column("role", sa.String(), server_default="attendee"),
        sa.Column("rsvp_status", sa.String(), server_default="pending"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("idx_meeting_participants_meeting_id", "meeting_participants", ["meeting_id"])

    op.create_table(
        "meeting_recordings",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("meeting_id", sa.String(), sa.ForeignKey("meetings.id"), nullable=False),
        sa.Column("recording_url", sa.Text(), nullable=True),
        sa.Column("recording_platform", sa.String(), nullable=True),
        sa.Column("duration_seconds", sa.Integer(), nullable=True),
        sa.Column("file_size_bytes", sa.BigInteger(), nullable=True),
        sa.Column("transcription_status", sa.String(), server_default="pending"),
        sa.Column("transcription_text", sa.Text(), nullable=True),
        sa.Column("transcription_segments", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("idx_meeting_recordings_meeting_id", "meeting_recordings", ["meeting_id"])

    op.create_table(
        "meeting_summaries",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("meeting_id", sa.String(), sa.ForeignKey("meetings.id"), nullable=False),
        sa.Column("recording_id", sa.String(), sa.ForeignKey("meeting_recordings.id"), nullable=True),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("key_decisions", postgresql.JSONB(), server_default=sa.text("'[]'::jsonb")),
        sa.Column("action_items", postgresql.JSONB(), server_default=sa.text("'[]'::jsonb")),
        sa.Column("follow_ups", postgresql.JSONB(), server_default=sa.text("'[]'::jsonb")),
        sa.Column("sentiment_analysis", postgresql.JSONB(), nullable=True),
        sa.Column("topics_discussed", postgresql.JSONB(), server_default=sa.text("'[]'::jsonb")),
        sa.Column("next_steps", sa.Text(), nullable=True),
        sa.Column("generated_by", sa.String(), server_default="ai"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("idx_meeting_summaries_meeting_id", "meeting_summaries", ["meeting_id"])

    op.create_table(
        "meeting_templates",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("user_id", sa.String(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("meeting_type", sa.String(), nullable=True),
        sa.Column("default_agenda", postgresql.JSONB(), server_default=sa.text("'[]'::jsonb")),
        sa.Column("default_duration_minutes", sa.Integer(), server_default="30"),
        sa.Column("pre_meeting_prompts", postgresql.JSONB(), server_default=sa.text("'[]'::jsonb")),
        sa.Column("post_meeting_prompts", postgresql.JSONB(), server_default=sa.text("'[]'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("idx_meeting_templates_user_id", "meeting_templates", ["user_id"])

    # ===== NOTES MODULE =====

    op.create_table(
        "notes",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("user_id", sa.String(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("content", sa.Text(), nullable=True),
        sa.Column("content_html", sa.Text(), nullable=True),
        sa.Column("note_type", sa.String(), server_default="idea"),
        sa.Column("linked_meeting_id", sa.String(), sa.ForeignKey("meetings.id"), nullable=True),
        sa.Column("linked_conversation_id", sa.String(), sa.ForeignKey("conversations.id"), nullable=True),
        sa.Column("tags", postgresql.JSONB(), server_default=sa.text("'[]'::jsonb")),
        sa.Column("is_pinned", sa.Boolean(), server_default="false"),
        sa.Column("startup_stage", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("idx_notes_user_id", "notes", ["user_id"])
    op.create_index("idx_notes_note_type", "notes", ["note_type"])

    op.create_table(
        "note_versions",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("note_id", sa.String(), sa.ForeignKey("notes.id"), nullable=False),
        sa.Column("version_number", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=True),
        sa.Column("content_html", sa.Text(), nullable=True),
        sa.Column("change_summary", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("idx_note_versions_note_id", "note_versions", ["note_id"])

    op.create_table(
        "note_links",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("source_note_id", sa.String(), sa.ForeignKey("notes.id"), nullable=False),
        sa.Column("target_note_id", sa.String(), sa.ForeignKey("notes.id"), nullable=False),
        sa.Column("link_type", sa.String(), server_default="related"),
        sa.Column("ai_generated", sa.Boolean(), server_default="false"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("idx_note_links_source", "note_links", ["source_note_id"])
    op.create_index("idx_note_links_target", "note_links", ["target_note_id"])

    # ===== CALENDAR MODULE =====

    op.create_table(
        "calendar_events",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("user_id", sa.String(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("event_type", sa.String(), server_default="meeting"),
        sa.Column("start_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("end_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("all_day", sa.Boolean(), server_default="false"),
        sa.Column("location", sa.Text(), nullable=True),
        sa.Column("meeting_link", sa.Text(), nullable=True),
        sa.Column("linked_meeting_id", sa.String(), sa.ForeignKey("meetings.id"), nullable=True),
        sa.Column("recurrence_rule", sa.Text(), nullable=True),
        sa.Column("reminders", postgresql.JSONB(), server_default=sa.text("'[]'::jsonb")),
        sa.Column("color", sa.String(), nullable=True),
        sa.Column("external_calendar_id", sa.String(), nullable=True),
        sa.Column("external_source", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("idx_calendar_events_user_id", "calendar_events", ["user_id"])
    op.create_index("idx_calendar_events_start_time", "calendar_events", ["start_time"])

    op.create_table(
        "calendar_integrations",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("user_id", sa.String(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("provider", sa.String(), nullable=False),
        sa.Column("access_token_encrypted", sa.Text(), nullable=True),
        sa.Column("refresh_token_encrypted", sa.Text(), nullable=True),
        sa.Column("calendar_ids", postgresql.JSONB(), server_default=sa.text("'[]'::jsonb")),
        sa.Column("sync_direction", sa.String(), server_default="import"),
        sa.Column("last_synced_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_active", sa.Boolean(), server_default="true"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("idx_calendar_integrations_user_id", "calendar_integrations", ["user_id"])

    op.create_table(
        "scheduling_preferences",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("user_id", sa.String(), sa.ForeignKey("users.id"), nullable=False, unique=True),
        sa.Column("working_hours", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb")),
        sa.Column("timezone", sa.String(), server_default="UTC"),
        sa.Column("focus_blocks", postgresql.JSONB(), server_default=sa.text("'[]'::jsonb")),
        sa.Column("meeting_buffer_minutes", sa.Integer(), server_default="15"),
        sa.Column("max_meetings_per_day", sa.Integer(), nullable=True),
        sa.Column("preferred_meeting_durations", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb")),
        sa.Column("scheduling_link", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )

    # ===== TRAVEL MODULE =====

    op.create_table(
        "trips",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("user_id", sa.String(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("purpose", sa.String(), server_default="other"),
        sa.Column("destination_city", sa.String(), nullable=True),
        sa.Column("destination_country", sa.String(), nullable=True),
        sa.Column("start_date", sa.Date(), nullable=True),
        sa.Column("end_date", sa.Date(), nullable=True),
        sa.Column("status", sa.String(), server_default="planning"),
        sa.Column("budget_cents", sa.BigInteger(), nullable=True),
        sa.Column("currency", sa.String(), server_default="USD"),
        sa.Column("linked_meeting_ids", postgresql.JSONB(), server_default=sa.text("'[]'::jsonb")),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("startup_stage", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("idx_trips_user_id", "trips", ["user_id"])

    op.create_table(
        "trip_items",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("trip_id", sa.String(), sa.ForeignKey("trips.id"), nullable=False),
        sa.Column("item_type", sa.String(), server_default="other"),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("start_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("end_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("location", sa.Text(), nullable=True),
        sa.Column("address", sa.Text(), nullable=True),
        sa.Column("confirmation_number", sa.String(), nullable=True),
        sa.Column("cost_cents", sa.BigInteger(), nullable=True),
        sa.Column("currency", sa.String(), server_default="USD"),
        sa.Column("booking_url", sa.Text(), nullable=True),
        sa.Column("status", sa.String(), server_default="researching"),
        sa.Column("sort_order", sa.Integer(), server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("idx_trip_items_trip_id", "trip_items", ["trip_id"])

    op.create_table(
        "travel_checklists",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("trip_id", sa.String(), sa.ForeignKey("trips.id"), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("items", postgresql.JSONB(), server_default=sa.text("'[]'::jsonb")),
        sa.Column("ai_generated", sa.Boolean(), server_default="false"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("idx_travel_checklists_trip_id", "travel_checklists", ["trip_id"])

    # ===== JOURNEY MODULE (new tables) =====

    op.create_table(
        "journey_stages",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("user_id", sa.String(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("startup_id", sa.String(), sa.ForeignKey("startups.id"), nullable=False),
        sa.Column("current_stage", sa.String(), nullable=False),
        sa.Column("stage_started_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("stage_completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("health_score", sa.Integer(), nullable=True),
        sa.Column("blockers", postgresql.JSONB(), server_default=sa.text("'[]'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("idx_journey_stages_user_id", "journey_stages", ["user_id"])
    op.create_index("idx_journey_stages_startup_id", "journey_stages", ["startup_id"])

    op.create_table(
        "journey_milestones",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("user_id", sa.String(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("stage", sa.String(), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("category", sa.String(), server_default="product"),
        sa.Column("status", sa.String(), server_default="not_started"),
        sa.Column("target_date", sa.Date(), nullable=True),
        sa.Column("completed_date", sa.Date(), nullable=True),
        sa.Column("evidence", postgresql.JSONB(), server_default=sa.text("'[]'::jsonb")),
        sa.Column("ai_suggestions", postgresql.JSONB(), server_default=sa.text("'[]'::jsonb")),
        sa.Column("sort_order", sa.Integer(), server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("idx_journey_milestones_user_id", "journey_milestones", ["user_id"])
    op.create_index("idx_journey_milestones_stage", "journey_milestones", ["stage"])

    op.create_table(
        "journey_playbooks",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("stage", sa.String(), nullable=False),
        sa.Column("category", sa.String(), nullable=True),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("tasks", postgresql.JSONB(), server_default=sa.text("'[]'::jsonb")),
        sa.Column("frameworks", postgresql.JSONB(), server_default=sa.text("'[]'::jsonb")),
        sa.Column("common_mistakes", postgresql.JSONB(), server_default=sa.text("'[]'::jsonb")),
        sa.Column("success_criteria", postgresql.JSONB(), server_default=sa.text("'[]'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("idx_journey_playbooks_stage", "journey_playbooks", ["stage"])

    # ===== CONTACTS MODULE =====

    op.create_table(
        "contacts",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("user_id", sa.String(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("email", sa.String(), nullable=True),
        sa.Column("phone", sa.String(), nullable=True),
        sa.Column("contact_type", sa.String(), server_default="other"),
        sa.Column("company", sa.String(), nullable=True),
        sa.Column("title", sa.String(), nullable=True),
        sa.Column("linkedin_url", sa.String(), nullable=True),
        sa.Column("twitter_url", sa.String(), nullable=True),
        sa.Column("relationship_strength", sa.String(), server_default="cold"),
        sa.Column("tags", postgresql.JSONB(), server_default=sa.text("'[]'::jsonb")),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("last_contacted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("next_follow_up_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("idx_contacts_user_id", "contacts", ["user_id"])
    op.create_index("idx_contacts_contact_type", "contacts", ["contact_type"])

    op.create_table(
        "contact_interactions",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("contact_id", sa.String(), sa.ForeignKey("contacts.id"), nullable=False),
        sa.Column("user_id", sa.String(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("interaction_type", sa.String(), server_default="other"),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("linked_meeting_id", sa.String(), sa.ForeignKey("meetings.id"), nullable=True),
        sa.Column("linked_note_id", sa.String(), sa.ForeignKey("notes.id"), nullable=True),
        sa.Column("occurred_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("idx_contact_interactions_contact_id", "contact_interactions", ["contact_id"])
    op.create_index("idx_contact_interactions_user_id", "contact_interactions", ["user_id"])


def downgrade() -> None:
    op.drop_table("contact_interactions")
    op.drop_table("contacts")
    op.drop_table("journey_playbooks")
    op.drop_table("journey_milestones")
    op.drop_table("journey_stages")
    op.drop_table("travel_checklists")
    op.drop_table("trip_items")
    op.drop_table("trips")
    op.drop_table("scheduling_preferences")
    op.drop_table("calendar_integrations")
    op.drop_table("calendar_events")
    op.drop_table("note_links")
    op.drop_table("note_versions")
    op.drop_table("notes")
    op.drop_table("meeting_templates")
    op.drop_table("meeting_summaries")
    op.drop_table("meeting_recordings")
    op.drop_table("meeting_participants")
    op.drop_table("meetings")
