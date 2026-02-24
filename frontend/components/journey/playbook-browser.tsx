"use client";

import { useEffect, useState } from "react";
import { apiFetch } from "@/lib/api";
import { JourneyPlaybook } from "@/lib/types";
import { Badge } from "@/components/ui/badge";
import { ChevronDown, ChevronRight, BookOpen, CheckSquare, Lightbulb, AlertTriangle } from "lucide-react";

interface PlaybookBrowserProps {
  currentStage: string;
}

export function PlaybookBrowser({ currentStage }: PlaybookBrowserProps) {
  const [playbooks, setPlaybooks] = useState<JourneyPlaybook[]>([]);
  const [expandedId, setExpandedId] = useState<string | null>(null);

  useEffect(() => {
    if (currentStage) {
      apiFetch<JourneyPlaybook[]>(`/journey/playbooks/?stage=${currentStage}`).then(setPlaybooks);
    }
  }, [currentStage]);

  function toggleExpand(id: string) {
    setExpandedId(expandedId === id ? null : id);
  }

  return (
    <div>
      <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-4">
        Playbooks
      </h3>

      {playbooks.length === 0 ? (
        <p className="text-sm text-muted-foreground text-center py-8">
          No playbooks for the {currentStage} stage yet.
        </p>
      ) : (
        <div className="space-y-2">
          {playbooks.map((pb) => {
            const isExpanded = expandedId === pb.id;
            return (
              <div key={pb.id} className="rounded-lg border border-border bg-card overflow-hidden">
                <button
                  onClick={() => toggleExpand(pb.id)}
                  className="w-full text-left p-4 flex items-start gap-3 hover:bg-accent/30 transition-colors"
                >
                  {isExpanded ? (
                    <ChevronDown className="h-4 w-4 mt-0.5 shrink-0 text-muted-foreground" />
                  ) : (
                    <ChevronRight className="h-4 w-4 mt-0.5 shrink-0 text-muted-foreground" />
                  )}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <BookOpen className="h-4 w-4 text-primary" />
                      <span className="font-medium text-sm">{pb.title}</span>
                      {pb.category && (
                        <Badge variant="secondary" className="text-[10px]">
                          {pb.category}
                        </Badge>
                      )}
                    </div>
                    {pb.description && (
                      <p className="text-xs text-muted-foreground mt-1">{pb.description}</p>
                    )}
                  </div>
                </button>

                {isExpanded && (
                  <div className="px-4 pb-4 space-y-4 border-t border-border pt-3">
                    {pb.tasks && pb.tasks.length > 0 && (
                      <Section
                        icon={<CheckSquare className="h-3.5 w-3.5 text-blue-400" />}
                        title="Tasks"
                        items={pb.tasks.map((t) => (typeof t === "string" ? t : JSON.stringify(t)))}
                      />
                    )}
                    {pb.frameworks && pb.frameworks.length > 0 && (
                      <Section
                        icon={<Lightbulb className="h-3.5 w-3.5 text-yellow-400" />}
                        title="Frameworks"
                        items={pb.frameworks.map((f) => (typeof f === "string" ? f : JSON.stringify(f)))}
                      />
                    )}
                    {pb.common_mistakes && pb.common_mistakes.length > 0 && (
                      <Section
                        icon={<AlertTriangle className="h-3.5 w-3.5 text-red-400" />}
                        title="Common Mistakes"
                        items={pb.common_mistakes}
                      />
                    )}
                    {pb.success_criteria && pb.success_criteria.length > 0 && (
                      <Section
                        icon={<CheckSquare className="h-3.5 w-3.5 text-green-400" />}
                        title="Success Criteria"
                        items={pb.success_criteria}
                      />
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function Section({ icon, title, items }: { icon: React.ReactNode; title: string; items: string[] }) {
  return (
    <div>
      <div className="flex items-center gap-1.5 mb-1.5">
        {icon}
        <span className="text-xs font-medium text-muted-foreground">{title}</span>
      </div>
      <ul className="space-y-1 ml-5">
        {items.map((item, i) => (
          <li key={i} className="text-xs text-foreground/80 list-disc">{item}</li>
        ))}
      </ul>
    </div>
  );
}
