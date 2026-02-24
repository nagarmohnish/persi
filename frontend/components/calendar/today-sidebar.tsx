"use client";

import { useEffect, useState } from "react";
import { apiFetch } from "@/lib/api";
import { CalendarEvent } from "@/lib/types";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Plus, Clock } from "lucide-react";
import { format } from "date-fns";

interface TodaySidebarProps {
  onCreateEvent: () => void;
  onEventClick: (event: CalendarEvent) => void;
  refreshKey: number;
}

export function TodaySidebar({ onCreateEvent, onEventClick, refreshKey }: TodaySidebarProps) {
  const [events, setEvents] = useState<CalendarEvent[]>([]);

  useEffect(() => {
    apiFetch<CalendarEvent[]>("/calendar/today/").then(setEvents);
  }, [refreshKey]);

  return (
    <div className="w-72 border-l border-border flex flex-col h-full">
      <div className="p-3 border-b border-border flex items-center justify-between">
        <h2 className="text-sm font-semibold">Today</h2>
        <Button size="sm" variant="ghost" onClick={onCreateEvent}>
          <Plus className="h-4 w-4" />
        </Button>
      </div>
      <ScrollArea className="flex-1">
        <div className="p-3 space-y-2">
          {events.length === 0 ? (
            <p className="text-xs text-muted-foreground text-center py-8">No events today</p>
          ) : (
            events.map((event) => (
              <button
                key={event.id}
                onClick={() => onEventClick(event)}
                className="w-full text-left rounded-lg border border-border p-3 hover:bg-accent/30 transition-colors"
              >
                <div className="flex items-center gap-2">
                  <Clock className="h-3.5 w-3.5 text-muted-foreground" />
                  {event.start_time && (
                    <span className="text-xs font-medium">
                      {format(new Date(event.start_time), "h:mm a")}
                      {event.end_time && ` – ${format(new Date(event.end_time), "h:mm a")}`}
                    </span>
                  )}
                  {event.all_day && <span className="text-xs font-medium">All day</span>}
                </div>
                <p className="text-sm font-medium mt-1">{event.title}</p>
                <Badge variant="secondary" className="text-[10px] mt-1">
                  {event.event_type}
                </Badge>
              </button>
            ))
          )}
        </div>
      </ScrollArea>
    </div>
  );
}
