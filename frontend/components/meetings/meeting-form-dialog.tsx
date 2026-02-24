"use client";

import { useState, useEffect } from "react";
import { apiFetch } from "@/lib/api";
import { Meeting, MeetingTemplate } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useToast } from "@/hooks/use-toast";

const MEETING_TYPES = [
  "one_on_one", "team", "investor", "customer", "advisory", "demo", "standup", "other",
];

interface MeetingFormDialogProps {
  open: boolean;
  onClose: () => void;
  meeting?: Meeting | null;
  template?: MeetingTemplate | null;
  onSaved: () => void;
}

export function MeetingFormDialog({ open, onClose, meeting, template, onSaved }: MeetingFormDialogProps) {
  const { toast } = useToast();
  const [title, setTitle] = useState("");
  const [meetingType, setMeetingType] = useState("other");
  const [startTime, setStartTime] = useState("");
  const [durationMinutes, setDurationMinutes] = useState(30);
  const [description, setDescription] = useState("");
  const [location, setLocation] = useState("");
  const [meetingLink, setMeetingLink] = useState("");
  const [platform, setPlatform] = useState("");
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    if (meeting) {
      setTitle(meeting.title);
      setMeetingType(meeting.meeting_type);
      setStartTime(meeting.start_time ? meeting.start_time.slice(0, 16) : "");
      setDurationMinutes(meeting.duration_minutes || 30);
      setDescription(meeting.description || "");
      setLocation(meeting.location || "");
      setMeetingLink(meeting.meeting_link || "");
      setPlatform(meeting.platform || "");
    } else if (template) {
      setTitle("");
      setMeetingType(template.meeting_type || "other");
      setDurationMinutes(template.default_duration_minutes || 30);
      setDescription(template.default_agenda?.map((a: unknown) => typeof a === "string" ? a : JSON.stringify(a)).join("\n") || "");
    } else {
      setTitle("");
      setMeetingType("other");
      setStartTime("");
      setDurationMinutes(30);
      setDescription("");
      setLocation("");
      setMeetingLink("");
      setPlatform("");
    }
  }, [meeting, template, open]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!title.trim()) return;
    setSubmitting(true);
    try {
      const body: Record<string, unknown> = {
        title: title.trim(),
        meeting_type: meetingType,
        duration_minutes: durationMinutes,
        description: description.trim() || undefined,
        location: location.trim() || undefined,
        meeting_link: meetingLink.trim() || undefined,
        platform: platform.trim() || undefined,
      };
      if (startTime) body.scheduled_at = new Date(startTime).toISOString();

      if (meeting) {
        await apiFetch(`/meetings/${meeting.id}`, {
          method: "PATCH",
          body: JSON.stringify(body),
        });
        toast({ title: "Meeting updated", description: "Your changes have been saved." });
      } else {
        await apiFetch("/meetings/", {
          method: "POST",
          body: JSON.stringify(body),
        });
        toast({ title: "Meeting created", description: "New meeting has been added." });
      }
      onSaved();
      onClose();
    } catch (err) {
      toast({ title: "Error", description: err instanceof Error ? err.message : "Something went wrong", variant: "destructive" });
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <Dialog open={open} onOpenChange={(o) => !o && onClose()}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{meeting ? "Edit Meeting" : "New Meeting"}</DialogTitle>
        </DialogHeader>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <Label>Title</Label>
            <Input value={title} onChange={(e) => setTitle(e.target.value)} placeholder="e.g., Investor pitch review" />
          </div>
          <div className="flex gap-3">
            <div className="flex-1">
              <Label>Type</Label>
              <Select value={meetingType} onValueChange={setMeetingType}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  {MEETING_TYPES.map((t) => (
                    <SelectItem key={t} value={t}>{t.replace(/_/g, " ")}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="flex-1">
              <Label>Duration (min)</Label>
              <Input type="number" value={durationMinutes} onChange={(e) => setDurationMinutes(Number(e.target.value))} />
            </div>
          </div>
          <div>
            <Label>Date & Time</Label>
            <Input type="datetime-local" value={startTime} onChange={(e) => setStartTime(e.target.value)} />
          </div>
          <div>
            <Label>Description / Agenda</Label>
            <Textarea value={description} onChange={(e) => setDescription(e.target.value)} rows={3} placeholder="Meeting agenda..." />
          </div>
          <div className="flex gap-3">
            <div className="flex-1">
              <Label>Location</Label>
              <Input value={location} onChange={(e) => setLocation(e.target.value)} placeholder="Office / Room" />
            </div>
            <div className="flex-1">
              <Label>Meeting Link</Label>
              <Input value={meetingLink} onChange={(e) => setMeetingLink(e.target.value)} placeholder="https://..." />
            </div>
          </div>
          <div className="flex justify-end gap-2">
            <Button type="button" variant="ghost" onClick={onClose}>Cancel</Button>
            <Button type="submit" disabled={submitting || !title.trim()}>
              {submitting ? "Saving..." : meeting ? "Update" : "Create"}
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
}
