"use client";

import { useEffect, useState, useCallback } from "react";
import { apiFetch } from "@/lib/api";
import { Meeting } from "@/lib/types";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Plus, Video } from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";
import { format } from "date-fns";

const STATUSES = [
  { value: "all", label: "All Status" },
  { value: "scheduled", label: "Scheduled" },
  { value: "in_progress", label: "In Progress" },
  { value: "completed", label: "Completed" },
  { value: "cancelled", label: "Cancelled" },
];

const TYPES = [
  { value: "all", label: "All Types" },
  { value: "one_on_one", label: "1:1" },
  { value: "team", label: "Team" },
  { value: "investor", label: "Investor" },
  { value: "customer", label: "Customer" },
  { value: "advisory", label: "Advisory" },
  { value: "demo", label: "Demo" },
  { value: "standup", label: "Standup" },
];

const STATUS_COLORS: Record<string, string> = {
  scheduled: "bg-blue-500/10 text-blue-400 border-blue-500/20",
  in_progress: "bg-yellow-500/10 text-yellow-400 border-yellow-500/20",
  completed: "bg-green-500/10 text-green-400 border-green-500/20",
  cancelled: "bg-red-500/10 text-red-400 border-red-500/20",
};

interface MeetingListProps {
  selectedId: string | null;
  onSelect: (id: string) => void;
  onNewMeeting: () => void;
}

export function MeetingList({ selectedId, onSelect, onNewMeeting }: MeetingListProps) {
  const [meetings, setMeetings] = useState<Meeting[]>([]);
  const [loading, setLoading] = useState(true);
  const [statusFilter, setStatusFilter] = useState("all");
  const [typeFilter, setTypeFilter] = useState("all");

  const loadMeetings = useCallback(async () => {
    const params = new URLSearchParams();
    if (statusFilter !== "all") params.set("status", statusFilter);
    if (typeFilter !== "all") params.set("meeting_type", typeFilter);
    const query = params.toString();
    const data = await apiFetch<Meeting[]>(`/meetings/${query ? `?${query}` : ""}`);
    setMeetings(data);
    setLoading(false);
  }, [statusFilter, typeFilter]);

  useEffect(() => {
    loadMeetings();
  }, [loadMeetings]);

  return (
    <div className="w-80 border-r border-border flex flex-col h-full">
      <div className="p-3 border-b border-border space-y-2">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold">Meetings</h2>
          <Button size="sm" variant="ghost" onClick={onNewMeeting}>
            <Plus className="h-4 w-4" />
          </Button>
        </div>
        <div className="flex gap-1">
          <Select value={statusFilter} onValueChange={setStatusFilter}>
            <SelectTrigger className="h-7 text-xs flex-1">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {STATUSES.map((s) => (
                <SelectItem key={s.value} value={s.value}>{s.label}</SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Select value={typeFilter} onValueChange={setTypeFilter}>
            <SelectTrigger className="h-7 text-xs flex-1">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {TYPES.map((t) => (
                <SelectItem key={t.value} value={t.value}>{t.label}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>
      <ScrollArea className="flex-1">
        <div className="p-2 space-y-1">
          {loading && Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="px-3 py-2.5 space-y-2">
              <Skeleton className="h-4 w-3/4" />
              <Skeleton className="h-3 w-1/2" />
            </div>
          ))}
          {!loading && meetings.map((m) => (
            <button
              key={m.id}
              onClick={() => onSelect(m.id)}
              className={`w-full text-left rounded-lg px-3 py-2.5 text-sm transition-colors ${
                m.id === selectedId ? "bg-accent text-accent-foreground" : "hover:bg-accent/50"
              }`}
            >
              <div className="flex items-center gap-2">
                <Video className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
                <span className="truncate font-medium">{m.title}</span>
              </div>
              <div className="flex items-center gap-1.5 mt-1 ml-5">
                <Badge variant="outline" className={`text-[10px] px-1 py-0 ${STATUS_COLORS[m.status] || ""}`}>
                  {m.status}
                </Badge>
                <Badge variant="secondary" className="text-[10px] px-1 py-0">
                  {m.meeting_type}
                </Badge>
                {m.start_time && (
                  <span className="text-[10px] text-muted-foreground">
                    {format(new Date(m.start_time), "MMM d, h:mm a")}
                  </span>
                )}
              </div>
            </button>
          ))}
          {!loading && meetings.length === 0 && (
            <p className="text-xs text-muted-foreground text-center py-8">No meetings found</p>
          )}
        </div>
      </ScrollArea>
    </div>
  );
}
