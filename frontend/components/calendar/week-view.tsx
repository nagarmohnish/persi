"use client";

import { useEffect, useState } from "react";
import { apiFetch } from "@/lib/api";
import { CalendarEvent } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { ChevronLeft, ChevronRight } from "lucide-react";
import {
  startOfWeek,
  endOfWeek,
  addWeeks,
  eachDayOfInterval,
  format,
  isSameDay,
  isToday,
} from "date-fns";

const TYPE_COLORS: Record<string, string> = {
  meeting: "bg-blue-500/80",
  call: "bg-green-500/80",
  deadline: "bg-red-500/80",
  reminder: "bg-yellow-500/80",
  personal: "bg-purple-500/80",
  focus_time: "bg-orange-500/80",
};

interface WeekViewProps {
  onEventClick: (event: CalendarEvent) => void;
  onDayClick: (date: Date) => void;
  refreshKey: number;
}

export function WeekView({ onEventClick, onDayClick, refreshKey }: WeekViewProps) {
  const [currentDate, setCurrentDate] = useState(new Date());
  const [events, setEvents] = useState<CalendarEvent[]>([]);

  const weekStart = startOfWeek(currentDate, { weekStartsOn: 1 });
  const weekEnd = endOfWeek(currentDate, { weekStartsOn: 1 });
  const days = eachDayOfInterval({ start: weekStart, end: weekEnd });

  useEffect(() => {
    const start = weekStart.toISOString();
    const end = weekEnd.toISOString();
    apiFetch<CalendarEvent[]>(`/calendar/events/?start=${start}&end=${end}`).then(setEvents);
  }, [currentDate, refreshKey]);

  function eventsForDay(day: Date) {
    return events.filter((e) => e.start_time && isSameDay(new Date(e.start_time), day));
  }

  return (
    <div className="flex-1 flex flex-col h-full">
      {/* Navigation */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border">
        <div className="flex items-center gap-2">
          <Button size="sm" variant="ghost" onClick={() => setCurrentDate(addWeeks(currentDate, -1))}>
            <ChevronLeft className="h-4 w-4" />
          </Button>
          <Button size="sm" variant="ghost" onClick={() => setCurrentDate(addWeeks(currentDate, 1))}>
            <ChevronRight className="h-4 w-4" />
          </Button>
          <Button size="sm" variant="outline" onClick={() => setCurrentDate(new Date())}>
            Today
          </Button>
        </div>
        <h2 className="text-sm font-semibold">
          {format(weekStart, "MMM d")} – {format(weekEnd, "MMM d, yyyy")}
        </h2>
      </div>

      {/* Week Grid */}
      <div className="flex-1 grid grid-cols-7 divide-x divide-border overflow-hidden">
        {days.map((day) => {
          const dayEvents = eventsForDay(day);
          const today = isToday(day);
          return (
            <div
              key={day.toISOString()}
              className="flex flex-col overflow-hidden cursor-pointer hover:bg-accent/20"
              onClick={() => onDayClick(day)}
            >
              {/* Day Header */}
              <div className={`px-2 py-2 text-center border-b border-border ${today ? "bg-primary/10" : ""}`}>
                <div className="text-[10px] text-muted-foreground uppercase">
                  {format(day, "EEE")}
                </div>
                <div className={`text-sm font-semibold ${today ? "text-primary" : ""}`}>
                  {format(day, "d")}
                </div>
              </div>

              {/* Events */}
              <div className="flex-1 p-1 space-y-1 overflow-y-auto">
                {dayEvents.map((event) => (
                  <button
                    key={event.id}
                    onClick={(e) => { e.stopPropagation(); onEventClick(event); }}
                    className={`w-full text-left rounded px-1.5 py-1 text-[10px] text-white truncate ${
                      TYPE_COLORS[event.event_type] || "bg-muted-foreground/60"
                    }`}
                  >
                    {event.start_time && (
                      <span className="font-medium">{format(new Date(event.start_time), "h:mm a")} </span>
                    )}
                    {event.title}
                  </button>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
