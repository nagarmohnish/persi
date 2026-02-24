"use client";

import { CalendarEvent, Meeting, JourneyMilestone, Note, JourneyStage } from "@/lib/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { FollowUpReminders } from "@/components/contacts/follow-up-reminders";
import { Clock, Video, Target, StickyNote, TrendingUp } from "lucide-react";
import { format } from "date-fns";
import Link from "next/link";

export function TodayScheduleWidget({ events }: { events: CalendarEvent[] }) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm flex items-center gap-2">
          <Clock className="h-4 w-4 text-muted-foreground" />
          Today&apos;s Schedule
          <Link href="/calendar" className="ml-auto text-xs text-primary hover:underline font-normal">View all</Link>
        </CardTitle>
      </CardHeader>
      <CardContent>
        {events.length === 0 ? (
          <p className="text-xs text-muted-foreground">No events today</p>
        ) : (
          <div className="space-y-2">
            {events.slice(0, 5).map((e) => (
              <div key={e.id} className="flex items-center gap-2 text-sm">
                <span className="text-xs text-muted-foreground w-16 shrink-0">
                  {e.start_time ? format(new Date(e.start_time), "h:mm a") : "All day"}
                </span>
                <span className="truncate">{e.title}</span>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export function UpcomingMeetingsWidget({ meetings }: { meetings: Meeting[] }) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm flex items-center gap-2">
          <Video className="h-4 w-4 text-muted-foreground" />
          Upcoming Meetings
          <Link href="/meetings" className="ml-auto text-xs text-primary hover:underline font-normal">View all</Link>
        </CardTitle>
      </CardHeader>
      <CardContent>
        {meetings.length === 0 ? (
          <p className="text-xs text-muted-foreground">No upcoming meetings</p>
        ) : (
          <div className="space-y-2">
            {meetings.slice(0, 5).map((m) => (
              <div key={m.id} className="flex items-center gap-2 text-sm">
                <Badge variant="secondary" className="text-[10px] shrink-0">{m.meeting_type}</Badge>
                <span className="truncate">{m.title}</span>
                {m.start_time && (
                  <span className="text-xs text-muted-foreground ml-auto shrink-0">
                    {format(new Date(m.start_time), "MMM d")}
                  </span>
                )}
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export function ActiveMilestonesWidget({ milestones }: { milestones: JourneyMilestone[] }) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm flex items-center gap-2">
          <Target className="h-4 w-4 text-muted-foreground" />
          Active Milestones
          <Link href="/journey" className="ml-auto text-xs text-primary hover:underline font-normal">View all</Link>
        </CardTitle>
      </CardHeader>
      <CardContent>
        {milestones.length === 0 ? (
          <p className="text-xs text-muted-foreground">No active milestones</p>
        ) : (
          <div className="space-y-2">
            {milestones.slice(0, 5).map((m) => (
              <div key={m.id} className="flex items-center gap-2 text-sm">
                <Badge variant="outline" className="text-[10px] shrink-0">{m.category}</Badge>
                <span className="truncate">{m.title}</span>
                <Badge variant="secondary" className="text-[10px] ml-auto">{m.status}</Badge>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export function RecentNotesWidget({ notes }: { notes: Note[] }) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm flex items-center gap-2">
          <StickyNote className="h-4 w-4 text-muted-foreground" />
          Recent Notes
          <Link href="/notes" className="ml-auto text-xs text-primary hover:underline font-normal">View all</Link>
        </CardTitle>
      </CardHeader>
      <CardContent>
        {notes.length === 0 ? (
          <p className="text-xs text-muted-foreground">No notes yet</p>
        ) : (
          <div className="space-y-2">
            {notes.slice(0, 5).map((n) => (
              <div key={n.id} className="flex items-center gap-2 text-sm">
                <Badge variant="secondary" className="text-[10px] shrink-0">{n.note_type}</Badge>
                <span className="truncate">{n.title}</span>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export function FollowUpsWidget() {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm flex items-center gap-2">
          <Clock className="h-4 w-4 text-muted-foreground" />
          Follow-up Reminders
          <Link href="/contacts" className="ml-auto text-xs text-primary hover:underline font-normal">View all</Link>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <FollowUpReminders limit={5} />
      </CardContent>
    </Card>
  );
}

export function StageHealthWidget({ stage }: { stage: JourneyStage | null }) {
  const healthScore = stage?.health_score;
  const healthColor = healthScore == null ? "bg-muted" : healthScore >= 70 ? "bg-green-500" : healthScore >= 30 ? "bg-yellow-500" : "bg-red-500";

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm flex items-center gap-2">
          <TrendingUp className="h-4 w-4 text-muted-foreground" />
          Stage Health
          <Link href="/journey" className="ml-auto text-xs text-primary hover:underline font-normal">View all</Link>
        </CardTitle>
      </CardHeader>
      <CardContent>
        {stage ? (
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-lg font-bold capitalize">{stage.current_stage}</span>
              {healthScore != null && <span className="text-sm font-semibold">{healthScore}/100</span>}
            </div>
            {healthScore != null && (
              <div className="w-full h-2 rounded-full bg-muted overflow-hidden">
                <div className={`h-full rounded-full ${healthColor}`} style={{ width: `${healthScore}%` }} />
              </div>
            )}
            {stage.blockers && stage.blockers.length > 0 && (
              <p className="text-xs text-yellow-400">{stage.blockers.length} blocker(s)</p>
            )}
          </div>
        ) : (
          <p className="text-xs text-muted-foreground">No stage set</p>
        )}
      </CardContent>
    </Card>
  );
}
