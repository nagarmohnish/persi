"use client";

import { useState, useEffect } from "react";
import { apiFetch } from "@/lib/api";
import { CalendarEvent } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
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

const EVENT_TYPES = ["meeting", "call", "deadline", "reminder", "personal", "focus_time"];

interface EventFormDialogProps {
  open: boolean;
  onClose: () => void;
  event?: CalendarEvent | null;
  defaultDate?: Date | null;
  onSaved: () => void;
}

export function EventFormDialog({ open, onClose, event, defaultDate, onSaved }: EventFormDialogProps) {
  const { toast } = useToast();
  const [title, setTitle] = useState("");
  const [eventType, setEventType] = useState("meeting");
  const [startTime, setStartTime] = useState("");
  const [endTime, setEndTime] = useState("");
  const [allDay, setAllDay] = useState(false);
  const [location, setLocation] = useState("");
  const [meetingLink, setMeetingLink] = useState("");
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    if (event) {
      setTitle(event.title);
      setEventType(event.event_type);
      setStartTime(event.start_time ? event.start_time.slice(0, 16) : "");
      setEndTime(event.end_time ? event.end_time.slice(0, 16) : "");
      setAllDay(event.all_day);
      setLocation(event.location || "");
      setMeetingLink(event.meeting_link || "");
    } else {
      setTitle("");
      setEventType("meeting");
      setStartTime(defaultDate ? `${defaultDate.toISOString().slice(0, 10)}T09:00` : "");
      setEndTime(defaultDate ? `${defaultDate.toISOString().slice(0, 10)}T10:00` : "");
      setAllDay(false);
      setLocation("");
      setMeetingLink("");
    }
  }, [event, defaultDate, open]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!title.trim()) return;
    setSubmitting(true);
    try {
      const body = {
        title: title.trim(),
        event_type: eventType,
        start_time: startTime ? new Date(startTime).toISOString() : undefined,
        end_time: endTime ? new Date(endTime).toISOString() : undefined,
        all_day: allDay,
        location: location.trim() || undefined,
        meeting_link: meetingLink.trim() || undefined,
      };
      if (event) {
        await apiFetch(`/calendar/events/${event.id}`, { method: "PATCH", body: JSON.stringify(body) });
        toast({ title: "Event updated", description: "Your changes have been saved." });
      } else {
        await apiFetch("/calendar/events/", { method: "POST", body: JSON.stringify(body) });
        toast({ title: "Event created", description: "New event has been added." });
      }
      onSaved();
      onClose();
    } catch (err) {
      toast({ title: "Error", description: err instanceof Error ? err.message : "Something went wrong", variant: "destructive" });
    } finally {
      setSubmitting(false);
    }
  }

  async function handleDelete() {
    if (!event) return;
    try {
      await apiFetch(`/calendar/events/${event.id}`, { method: "DELETE" });
      toast({ title: "Event deleted", description: "The event has been removed." });
      onSaved();
      onClose();
    } catch (err) {
      toast({ title: "Error", description: err instanceof Error ? err.message : "Failed to delete event", variant: "destructive" });
    }
  }

  return (
    <Dialog open={open} onOpenChange={(o) => !o && onClose()}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{event ? "Edit Event" : "New Event"}</DialogTitle>
        </DialogHeader>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <Label>Title</Label>
            <Input value={title} onChange={(e) => setTitle(e.target.value)} placeholder="e.g., Team standup" />
          </div>
          <div className="flex gap-3">
            <div className="flex-1">
              <Label>Type</Label>
              <Select value={eventType} onValueChange={setEventType}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  {EVENT_TYPES.map((t) => (
                    <SelectItem key={t} value={t}>{t.replace(/_/g, " ")}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="flex items-end">
              <label className="flex items-center gap-2 text-sm pb-2 cursor-pointer">
                <Checkbox checked={allDay} onCheckedChange={(c) => setAllDay(!!c)} />
                All day
              </label>
            </div>
          </div>
          {!allDay && (
            <div className="flex gap-3">
              <div className="flex-1">
                <Label>Start</Label>
                <Input type="datetime-local" value={startTime} onChange={(e) => setStartTime(e.target.value)} />
              </div>
              <div className="flex-1">
                <Label>End</Label>
                <Input type="datetime-local" value={endTime} onChange={(e) => setEndTime(e.target.value)} />
              </div>
            </div>
          )}
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
          <div className="flex justify-between">
            {event && (
              <Button type="button" variant="destructive" size="sm" onClick={handleDelete}>
                Delete
              </Button>
            )}
            <div className="flex gap-2 ml-auto">
              <Button type="button" variant="ghost" onClick={onClose}>Cancel</Button>
              <Button type="submit" disabled={submitting || !title.trim()}>
                {submitting ? "Saving..." : event ? "Update" : "Create"}
              </Button>
            </div>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
}
