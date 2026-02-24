export interface Note {
  id: string;
  user_id: string;
  title: string;
  content?: string;
  content_html?: string;
  note_type: string;
  linked_meeting_id?: string;
  linked_conversation_id?: string;
  tags: string[];
  is_pinned: boolean;
  startup_stage?: string;
  created_at: string;
  updated_at: string;
}

export interface NoteVersion {
  id: string;
  note_id: string;
  version_number: number;
  content?: string;
  content_html?: string;
  change_summary?: string;
  created_at: string;
}

export interface Meeting {
  id: string;
  user_id: string;
  title: string;
  description?: string;
  meeting_type: string;
  status: string;
  start_time?: string;
  end_time?: string;
  duration_minutes?: number;
  location?: string;
  meeting_link?: string;
  platform?: string;
  is_recurring: boolean;
  startup_stage?: string;
  created_at: string;
  updated_at: string;
}

export interface MeetingParticipant {
  id: string;
  meeting_id: string;
  name: string;
  email?: string;
  role: string;
  rsvp_status: string;
  created_at: string;
}

export interface MeetingSummary {
  id: string;
  meeting_id: string;
  recording_id?: string;
  summary: string;
  key_decisions: unknown[];
  action_items: unknown[];
  follow_ups: unknown[];
  next_steps?: string;
  generated_by: string;
  created_at: string;
  updated_at: string;
}

export interface MeetingTemplate {
  id: string;
  user_id: string;
  name: string;
  meeting_type?: string;
  default_agenda: unknown[];
  default_duration_minutes: number;
  pre_meeting_prompts: unknown[];
  post_meeting_prompts: unknown[];
  created_at: string;
  updated_at: string;
}

export interface CalendarEvent {
  id: string;
  user_id: string;
  title: string;
  description?: string;
  event_type: string;
  start_time?: string;
  end_time?: string;
  all_day: boolean;
  location?: string;
  meeting_link?: string;
  linked_meeting_id?: string;
  recurrence_rule?: string;
  reminders: unknown[];
  color?: string;
  external_source?: string;
  created_at: string;
  updated_at: string;
}

export interface Contact {
  id: string;
  user_id: string;
  name: string;
  email?: string;
  phone?: string;
  contact_type: string;
  company?: string;
  title?: string;
  linkedin_url?: string;
  twitter_url?: string;
  relationship_strength: string;
  tags: string[];
  notes?: string;
  last_contacted_at?: string;
  next_follow_up_at?: string;
  created_at: string;
  updated_at: string;
}

export interface ContactInteraction {
  id: string;
  contact_id: string;
  user_id: string;
  interaction_type: string;
  summary?: string;
  linked_meeting_id?: string;
  linked_note_id?: string;
  occurred_at: string;
  created_at: string;
}

export interface Trip {
  id: string;
  user_id: string;
  title: string;
  purpose: string;
  destination_city?: string;
  destination_country?: string;
  start_date?: string;
  end_date?: string;
  status: string;
  budget_cents?: number;
  currency: string;
  notes?: string;
  created_at: string;
  updated_at: string;
}

export interface TripItem {
  id: string;
  trip_id: string;
  item_type: string;
  title: string;
  description?: string;
  start_time?: string;
  end_time?: string;
  location?: string;
  address?: string;
  confirmation_number?: string;
  cost_cents?: number;
  currency: string;
  booking_url?: string;
  status: string;
  sort_order: number;
  created_at: string;
  updated_at: string;
}

export interface JourneyStage {
  id: string;
  user_id: string;
  startup_id: string;
  current_stage: string;
  stage_started_at: string;
  stage_completed_at?: string;
  health_score?: number;
  blockers: Array<{ description: string; severity: string; suggested_action: string }>;
  created_at: string;
  updated_at: string;
}

export interface JourneyMilestone {
  id: string;
  user_id: string;
  stage: string;
  title: string;
  description?: string;
  category: string;
  status: string;
  target_date?: string;
  completed_date?: string;
  evidence: unknown[];
  ai_suggestions: unknown[];
  sort_order: number;
  created_at: string;
  updated_at: string;
}

export interface JourneyPlaybook {
  id: string;
  stage: string;
  category?: string;
  title: string;
  description?: string;
  tasks: unknown[];
  frameworks: unknown[];
  common_mistakes: string[];
  success_criteria: string[];
  created_at: string;
  updated_at: string;
}

export interface Startup {
  id: string;
  user_id: string;
  name?: string;
  one_liner?: string;
  problem_statement?: string;
  target_audience?: string;
  stage: string;
  industry?: string;
  business_model?: string;
  context_notes?: string;
  created_at: string;
  updated_at: string;
}

export interface User {
  id: string;
  email: string;
  full_name?: string;
  avatar_url?: string;
  onboarding_completed: boolean;
}

export interface Conversation {
  id: string;
  user_id: string;
  startup_id?: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface Message {
  id: string;
  conversation_id: string;
  role: "user" | "assistant" | "system" | "tool";
  content: string;
  created_at: string;
}
