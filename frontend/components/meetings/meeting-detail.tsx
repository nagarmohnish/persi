"use client";

import { useEffect, useState } from "react";
import { apiFetch } from "@/lib/api";
import { Meeting } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ParticipantList } from "./participant-list";
import { Video, Calendar, Clock, MapPin, Link2, Edit, Trash2, Sparkles } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { format } from "date-fns";

const STATUS_COLORS: Record<string, string> = {
  scheduled: "bg-blue-500/10 text-blue-400 border-blue-500/20",
  in_progress: "bg-yellow-500/10 text-yellow-400 border-yellow-500/20",
  completed: "bg-green-500/10 text-green-400 border-green-500/20",
  cancelled: "bg-red-500/10 text-red-400 border-red-500/20",
};

interface MeetingDetailProps {
  meetingId: string | null;
  onEdit: (meeting: Meeting) => void;
  onDeleted: () => void;
}

export function MeetingDetail({ meetingId, onEdit, onDeleted }: MeetingDetailProps) {
  const { toast } = useToast();
  const [meeting, setMeeting] = useState<Meeting | null>(null);
  const [summary, setSummary] = useState<string | null>(null);
  const [generatingSummary, setGeneratingSummary] = useState(false);

  useEffect(() => {
    if (!meetingId) { setMeeting(null); return; }
    apiFetch<Meeting>(`/meetings/${meetingId}`).then(setMeeting);
    // Try to load existing summary
    apiFetch<{ summary: string } | null>(`/meetings/${meetingId}/summary`)
      .then((s) => setSummary(s?.summary || null))
      .catch(() => setSummary(null));
  }, [meetingId]);

  async function handleDelete() {
    if (!meetingId) return;
    try {
      await apiFetch(`/meetings/${meetingId}`, { method: "DELETE" });
      toast({ title: "Meeting deleted", description: "The meeting has been removed." });
      onDeleted();
    } catch (err) {
      toast({ title: "Error", description: err instanceof Error ? err.message : "Failed to delete meeting", variant: "destructive" });
    }
  }

  async function handleGenerateSummary() {
    if (!meetingId) return;
    setGeneratingSummary(true);
    try {
      const result = await apiFetch<{ summary: string }>(`/meetings/${meetingId}/summary`, {
        method: "POST",
      });
      setSummary(result.summary);
      toast({ title: "Summary generated", description: "AI summary is ready." });
    } catch (err) {
      toast({ title: "Error", description: err instanceof Error ? err.message : "Failed to generate summary", variant: "destructive" });
    } finally {
      setGeneratingSummary(false);
    }
  }

  if (!meetingId || !meeting) {
    return (
      <div className="flex-1 flex items-center justify-center text-muted-foreground">
        <div className="text-center">
          <Video className="h-12 w-12 opacity-20 mx-auto mb-3" />
          <p className="text-sm">Select a meeting to view details</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="max-w-2xl mx-auto p-6 space-y-6">
        {/* Header */}
        <div className="flex items-start justify-between">
          <div>
            <h2 className="text-xl font-semibold">{meeting.title}</h2>
            <div className="flex items-center gap-2 mt-2">
              <Badge variant="outline" className={STATUS_COLORS[meeting.status] || ""}>
                {meeting.status}
              </Badge>
              <Badge variant="secondary">{meeting.meeting_type.replace(/_/g, " ")}</Badge>
            </div>
          </div>
          <div className="flex gap-1">
            <Button size="sm" variant="ghost" onClick={() => onEdit(meeting)}>
              <Edit className="h-4 w-4" />
            </Button>
            <Button size="sm" variant="ghost" className="text-destructive" onClick={handleDelete}>
              <Trash2 className="h-4 w-4" />
            </Button>
          </div>
        </div>

        {/* Info */}
        <div className="grid grid-cols-2 gap-3 text-sm">
          {meeting.start_time && (
            <div className="flex items-center gap-2 text-muted-foreground">
              <Calendar className="h-4 w-4" />
              {format(new Date(meeting.start_time), "EEE, MMM d, yyyy 'at' h:mm a")}
            </div>
          )}
          {meeting.duration_minutes && (
            <div className="flex items-center gap-2 text-muted-foreground">
              <Clock className="h-4 w-4" />
              {meeting.duration_minutes} minutes
            </div>
          )}
          {meeting.location && (
            <div className="flex items-center gap-2 text-muted-foreground">
              <MapPin className="h-4 w-4" />
              {meeting.location}
            </div>
          )}
          {meeting.meeting_link && (
            <div className="flex items-center gap-2 text-muted-foreground">
              <Link2 className="h-4 w-4" />
              <a href={meeting.meeting_link} target="_blank" rel="noopener noreferrer" className="text-primary hover:underline truncate">
                {meeting.meeting_link}
              </a>
            </div>
          )}
        </div>

        {/* Description */}
        {meeting.description && (
          <div>
            <h3 className="text-sm font-semibold mb-1">Description</h3>
            <p className="text-sm text-muted-foreground whitespace-pre-wrap">{meeting.description}</p>
          </div>
        )}

        {/* Participants */}
        <div className="border-t border-border pt-4">
          <ParticipantList meetingId={meeting.id} />
        </div>

        {/* AI Summary */}
        <div className="border-t border-border pt-4">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-sm font-semibold">AI Summary</h3>
            <Button
              size="sm"
              variant="outline"
              className="h-7 text-xs"
              onClick={handleGenerateSummary}
              disabled={generatingSummary}
            >
              <Sparkles className="h-3 w-3 mr-1" />
              {generatingSummary ? "Generating..." : "Generate Summary"}
            </Button>
          </div>
          {summary ? (
            <p className="text-sm text-muted-foreground whitespace-pre-wrap">{summary}</p>
          ) : (
            <p className="text-xs text-muted-foreground">No summary generated yet.</p>
          )}
        </div>
      </div>
    </div>
  );
}
