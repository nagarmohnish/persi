"use client";

import { useEffect, useState, useCallback } from "react";
import { apiFetch } from "@/lib/api";
import { JourneyStage, Startup } from "@/lib/types";
import { StageHeader } from "@/components/journey/stage-header";
import { MilestoneList } from "@/components/journey/milestone-list";
import { PlaybookBrowser } from "@/components/journey/playbook-browser";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";

export default function JourneyPage() {
  const [stage, setStage] = useState<JourneyStage | null>(null);
  const [startup, setStartup] = useState<Startup | null>(null);

  const loadData = useCallback(async () => {
    const [stageData, startupData] = await Promise.all([
      apiFetch<JourneyStage | null>("/journey/current/"),
      apiFetch<Startup | null>("/profile/startup"),
    ]);
    setStage(stageData);
    setStartup(startupData);
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const currentStage = stage?.current_stage || startup?.stage || "ideation";

  return (
    <ScrollArea className="h-full">
      <div className="max-w-4xl mx-auto p-6 space-y-6">
        <StageHeader stage={stage} startup={startup} onStageChanged={loadData} />

        <Tabs defaultValue="milestones">
          <TabsList>
            <TabsTrigger value="milestones">Milestones</TabsTrigger>
            <TabsTrigger value="playbooks">Playbooks</TabsTrigger>
          </TabsList>
          <TabsContent value="milestones" className="mt-4">
            <MilestoneList currentStage={currentStage} />
          </TabsContent>
          <TabsContent value="playbooks" className="mt-4">
            <PlaybookBrowser currentStage={currentStage} />
          </TabsContent>
        </Tabs>
      </div>
    </ScrollArea>
  );
}
