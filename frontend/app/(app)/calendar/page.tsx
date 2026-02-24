"use client";

import { useState } from "react";
import { CalendarEvent } from "@/lib/types";
import { WeekView } from "@/components/calendar/week-view";
import { TodaySidebar } from "@/components/calendar/today-sidebar";
import { EventFormDialog } from "@/components/calendar/event-form-dialog";

export default function CalendarPage() {
  const [showForm, setShowForm] = useState(false);
  const [editingEvent, setEditingEvent] = useState<CalendarEvent | null>(null);
  const [defaultDate, setDefaultDate] = useState<Date | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);

  function handleEventClick(event: CalendarEvent) {
    setEditingEvent(event);
    setDefaultDate(null);
    setShowForm(true);
  }

  function handleDayClick(date: Date) {
    setEditingEvent(null);
    setDefaultDate(date);
    setShowForm(true);
  }

  function handleCreateEvent() {
    setEditingEvent(null);
    setDefaultDate(new Date());
    setShowForm(true);
  }

  function handleSaved() {
    setRefreshKey((k) => k + 1);
  }

  return (
    <div className="flex h-full">
      <WeekView
        onEventClick={handleEventClick}
        onDayClick={handleDayClick}
        refreshKey={refreshKey}
      />
      <TodaySidebar
        onCreateEvent={handleCreateEvent}
        onEventClick={handleEventClick}
        refreshKey={refreshKey}
      />
      <EventFormDialog
        open={showForm}
        onClose={() => setShowForm(false)}
        event={editingEvent}
        defaultDate={defaultDate}
        onSaved={handleSaved}
      />
    </div>
  );
}
