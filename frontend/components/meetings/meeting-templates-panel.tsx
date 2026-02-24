"use client";

import { useEffect, useState } from "react";
import { apiFetch } from "@/lib/api";
import { MeetingTemplate } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { FileText, Plus, Play } from "lucide-react";

const MEETING_TYPES = ["one_on_one", "team", "investor", "customer", "advisory", "demo", "standup", "other"];

interface MeetingTemplatesPanelProps {
  onUseTemplate: (template: MeetingTemplate) => void;
}

export function MeetingTemplatesPanel({ onUseTemplate }: MeetingTemplatesPanelProps) {
  const [templates, setTemplates] = useState<MeetingTemplate[]>([]);
  const [showCreate, setShowCreate] = useState(false);
  const [name, setName] = useState("");
  const [meetingType, setMeetingType] = useState("other");
  const [duration, setDuration] = useState(30);

  async function load() {
    const data = await apiFetch<MeetingTemplate[]>("/meetings/templates/");
    setTemplates(data);
  }

  useEffect(() => {
    load();
  }, []);

  async function handleCreate(e: React.FormEvent) {
    e.preventDefault();
    if (!name.trim()) return;
    await apiFetch("/meetings/templates/", {
      method: "POST",
      body: JSON.stringify({
        name: name.trim(),
        meeting_type: meetingType,
        default_duration_minutes: duration,
      }),
    });
    setName("");
    setShowCreate(false);
    load();
  }

  return (
    <div className="border-t border-border p-3">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Templates</h3>
        <Button size="sm" variant="ghost" className="h-6 text-xs" onClick={() => setShowCreate(!showCreate)}>
          <Plus className="h-3 w-3 mr-1" /> New
        </Button>
      </div>

      {showCreate && (
        <form onSubmit={handleCreate} className="space-y-2 mb-3">
          <Input className="h-7 text-xs" placeholder="Template name" value={name} onChange={(e) => setName(e.target.value)} />
          <div className="flex gap-1.5">
            <Select value={meetingType} onValueChange={setMeetingType}>
              <SelectTrigger className="h-7 text-xs flex-1"><SelectValue /></SelectTrigger>
              <SelectContent>
                {MEETING_TYPES.map((t) => (
                  <SelectItem key={t} value={t}>{t.replace(/_/g, " ")}</SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Input className="h-7 text-xs w-16" type="number" value={duration} onChange={(e) => setDuration(Number(e.target.value))} />
            <Button type="submit" size="sm" className="h-7 text-xs">Save</Button>
          </div>
        </form>
      )}

      <div className="space-y-1">
        {templates.map((t) => (
          <div key={t.id} className="flex items-center gap-2 rounded px-2 py-1.5 text-sm hover:bg-accent/30">
            <FileText className="h-3.5 w-3.5 text-muted-foreground" />
            <span className="flex-1 truncate">{t.name}</span>
            <Badge variant="secondary" className="text-[10px]">{t.default_duration_minutes}m</Badge>
            <button onClick={() => onUseTemplate(t)} className="text-primary hover:text-primary/80">
              <Play className="h-3 w-3" />
            </button>
          </div>
        ))}
        {templates.length === 0 && !showCreate && (
          <p className="text-xs text-muted-foreground text-center py-2">No templates yet</p>
        )}
      </div>
    </div>
  );
}
