"use client";

import { JourneyStage, Startup } from "@/lib/types";
import { apiFetch } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { AlertTriangle, TrendingUp } from "lucide-react";
import { formatDistanceToNow } from "date-fns";
import { useState } from "react";

const STAGES = [
  { value: "ideation", label: "Ideation" },
  { value: "validation", label: "Validation" },
  { value: "mvp", label: "MVP" },
  { value: "launch", label: "Launch" },
  { value: "growth", label: "Growth" },
  { value: "fundraising", label: "Fundraising" },
  { value: "scaling", label: "Scaling" },
];

interface StageHeaderProps {
  stage: JourneyStage | null;
  startup: Startup | null;
  onStageChanged: () => void;
}

export function StageHeader({ stage, startup, onStageChanged }: StageHeaderProps) {
  const [newStage, setNewStage] = useState("");

  async function handleAdvance() {
    if (!newStage || !startup) return;
    await apiFetch(`/journey/stage/?startup_id=${startup.id}&stage=${newStage}`, {
      method: "POST",
    });
    setNewStage("");
    onStageChanged();
  }

  const healthScore = stage?.health_score;
  const healthColor = healthScore == null ? "bg-muted" : healthScore >= 70 ? "bg-green-500" : healthScore >= 30 ? "bg-yellow-500" : "bg-red-500";

  return (
    <div className="rounded-xl border border-border bg-card p-6">
      <div className="flex items-start justify-between">
        <div>
          <div className="flex items-center gap-3">
            <h2 className="text-2xl font-bold capitalize">
              {stage?.current_stage || "No stage set"}
            </h2>
            {stage && (
              <Badge variant="outline" className="capitalize">
                {formatDistanceToNow(new Date(stage.stage_started_at), { addSuffix: true })}
              </Badge>
            )}
          </div>
          {startup && (
            <p className="text-sm text-muted-foreground mt-1">{startup.name}</p>
          )}
        </div>

        {/* Health Score */}
        {healthScore != null && (
          <div className="text-right">
            <div className="flex items-center gap-2">
              <TrendingUp className="h-4 w-4 text-muted-foreground" />
              <span className="text-sm font-medium">Health Score</span>
            </div>
            <div className="flex items-center gap-2 mt-1">
              <div className="w-24 h-2 rounded-full bg-muted overflow-hidden">
                <div
                  className={`h-full rounded-full ${healthColor}`}
                  style={{ width: `${healthScore}%` }}
                />
              </div>
              <span className="text-sm font-semibold">{healthScore}</span>
            </div>
          </div>
        )}
      </div>

      {/* Stage Selector */}
      <div className="flex items-center gap-2 mt-4">
        <Select value={newStage} onValueChange={setNewStage}>
          <SelectTrigger className="w-48 h-8 text-sm">
            <SelectValue placeholder="Advance to stage..." />
          </SelectTrigger>
          <SelectContent>
            {STAGES.filter((s) => s.value !== stage?.current_stage).map((s) => (
              <SelectItem key={s.value} value={s.value}>
                {s.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Button size="sm" onClick={handleAdvance} disabled={!newStage}>
          Set Stage
        </Button>
      </div>

      {/* Blockers */}
      {stage?.blockers && stage.blockers.length > 0 && (
        <div className="mt-4 space-y-2">
          <h3 className="text-sm font-medium flex items-center gap-1.5">
            <AlertTriangle className="h-4 w-4 text-yellow-500" />
            Blockers
          </h3>
          {stage.blockers.map((b, i) => (
            <div key={i} className="rounded-lg bg-yellow-500/10 border border-yellow-500/20 p-3 text-sm">
              <p className="font-medium">{b.description}</p>
              {b.suggested_action && (
                <p className="text-muted-foreground text-xs mt-1">Suggestion: {b.suggested_action}</p>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
