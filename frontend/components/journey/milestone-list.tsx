"use client";

import { useEffect, useState } from "react";
import { apiFetch } from "@/lib/api";
import { JourneyMilestone } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { MilestoneForm } from "./milestone-form";
import { Plus, Circle, Clock, CheckCircle2 } from "lucide-react";
import { format } from "date-fns";

const STATUS_CYCLE = ["not_started", "in_progress", "completed"] as const;
const STATUS_CONFIG: Record<string, { icon: typeof Circle; color: string; label: string }> = {
  not_started: { icon: Circle, color: "text-muted-foreground", label: "Not Started" },
  in_progress: { icon: Clock, color: "text-yellow-500", label: "In Progress" },
  completed: { icon: CheckCircle2, color: "text-green-500", label: "Completed" },
};

const CATEGORY_COLORS: Record<string, string> = {
  product: "bg-blue-500/10 text-blue-400 border-blue-500/20",
  customer: "bg-purple-500/10 text-purple-400 border-purple-500/20",
  revenue: "bg-green-500/10 text-green-400 border-green-500/20",
  team: "bg-orange-500/10 text-orange-400 border-orange-500/20",
  funding: "bg-yellow-500/10 text-yellow-400 border-yellow-500/20",
  legal: "bg-red-500/10 text-red-400 border-red-500/20",
  operations: "bg-gray-500/10 text-gray-400 border-gray-500/20",
};

interface MilestoneListProps {
  currentStage: string;
}

export function MilestoneList({ currentStage }: MilestoneListProps) {
  const [milestones, setMilestones] = useState<JourneyMilestone[]>([]);
  const [showForm, setShowForm] = useState(false);

  async function loadMilestones() {
    const data = await apiFetch<JourneyMilestone[]>(
      `/journey/milestones/${currentStage ? `?stage=${currentStage}` : ""}`
    );
    setMilestones(data);
  }

  useEffect(() => {
    if (currentStage) loadMilestones();
  }, [currentStage]);

  async function handleStatusToggle(milestone: JourneyMilestone) {
    const currentIdx = STATUS_CYCLE.indexOf(milestone.status as typeof STATUS_CYCLE[number]);
    const nextStatus = STATUS_CYCLE[(currentIdx + 1) % STATUS_CYCLE.length];
    await apiFetch(`/journey/milestones/${milestone.id}`, {
      method: "PATCH",
      body: JSON.stringify({
        status: nextStatus,
        completed_date: nextStatus === "completed" ? new Date().toISOString().split("T")[0] : null,
      }),
    });
    loadMilestones();
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">
          Milestones
        </h3>
        <Button size="sm" variant="outline" onClick={() => setShowForm(true)}>
          <Plus className="h-3.5 w-3.5 mr-1" /> Add Milestone
        </Button>
      </div>

      {milestones.length === 0 ? (
        <p className="text-sm text-muted-foreground text-center py-8">
          No milestones for this stage yet. Add one to start tracking progress.
        </p>
      ) : (
        <div className="space-y-2">
          {milestones.map((m) => {
            const config = STATUS_CONFIG[m.status] || STATUS_CONFIG.not_started;
            const StatusIcon = config.icon;
            return (
              <div key={m.id} className="rounded-lg border border-border bg-card p-4">
                <div className="flex items-start gap-3">
                  <button onClick={() => handleStatusToggle(m)} className="mt-0.5">
                    <StatusIcon className={`h-5 w-5 ${config.color} transition-colors`} />
                  </button>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className={`font-medium text-sm ${m.status === "completed" ? "line-through text-muted-foreground" : ""}`}>
                        {m.title}
                      </span>
                      <Badge variant="outline" className={`text-[10px] ${CATEGORY_COLORS[m.category] || ""}`}>
                        {m.category}
                      </Badge>
                    </div>
                    {m.description && (
                      <p className="text-xs text-muted-foreground mt-1">{m.description}</p>
                    )}
                    {m.target_date && (
                      <p className="text-[10px] text-muted-foreground mt-1">
                        Target: {format(new Date(m.target_date), "MMM d, yyyy")}
                      </p>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      <MilestoneForm
        open={showForm}
        onClose={() => setShowForm(false)}
        currentStage={currentStage}
        onCreated={loadMilestones}
      />
    </div>
  );
}
