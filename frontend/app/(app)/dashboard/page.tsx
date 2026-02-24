"use client";

import { useEffect, useState } from "react";
import { apiFetch } from "@/lib/api";
import { CalendarEvent, Meeting, JourneyMilestone, Note, JourneyStage } from "@/lib/types";
import { QuickActions } from "@/components/dashboard/quick-actions";
import {
  TodayScheduleWidget,
  UpcomingMeetingsWidget,
  ActiveMilestonesWidget,
  RecentNotesWidget,
  FollowUpsWidget,
  StageHealthWidget,
} from "@/components/dashboard/dashboard-widgets";
import { ScrollArea } from "@/components/ui/scroll-area";

export default function DashboardPage() {
  const [todayEvents, setTodayEvents] = useState<CalendarEvent[]>([]);
  const [meetings, setMeetings] = useState<Meeting[]>([]);
  const [milestones, setMilestones] = useState<JourneyMilestone[]>([]);
  const [notes, setNotes] = useState<Note[]>([]);
  const [stage, setStage] = useState<JourneyStage | null>(null);

  useEffect(() => {
    Promise.allSettled([
      apiFetch<CalendarEvent[]>("/calendar/today/").then(setTodayEvents),
      apiFetch<Meeting[]>("/meetings/?status=scheduled").then(setMeetings),
      apiFetch<JourneyMilestone[]>("/journey/milestones/").then(setMilestones),
      apiFetch<Note[]>("/notes/").then(setNotes),
      apiFetch<JourneyStage | null>("/journey/current/").then(setStage),
    ]);
  }, []);

  const activeMilestones = milestones.filter((m) => m.status !== "completed");

  return (
    <ScrollArea className="h-full">
      <div className="p-6 space-y-6">
        {/* Quick Actions */}
        <div>
          <h1 className="text-2xl font-bold mb-4">Dashboard</h1>
          <QuickActions />
        </div>

        {/* Widget Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <TodayScheduleWidget events={todayEvents} />
          <UpcomingMeetingsWidget meetings={meetings} />
          <ActiveMilestonesWidget milestones={activeMilestones} />
          <RecentNotesWidget notes={notes} />
          <FollowUpsWidget />
          <StageHealthWidget stage={stage} />
        </div>
      </div>
    </ScrollArea>
  );
}
