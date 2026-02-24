"use client";

import { useEffect, useState } from "react";
import { apiFetch } from "@/lib/api";
import { ContactInteraction } from "@/lib/types";
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
import { Plus, Mail, Phone, Video, MessageSquare, Users, MoreHorizontal } from "lucide-react";
import { formatDistanceToNow } from "date-fns";

const INTERACTION_TYPES = ["email", "meeting", "call", "message", "introduction", "other"];
const TYPE_ICONS: Record<string, typeof Mail> = {
  email: Mail,
  call: Phone,
  meeting: Video,
  message: MessageSquare,
  introduction: Users,
  other: MoreHorizontal,
};

interface InteractionLogProps {
  contactId: string;
}

export function InteractionLog({ contactId }: InteractionLogProps) {
  const [interactions, setInteractions] = useState<ContactInteraction[]>([]);
  const [showAdd, setShowAdd] = useState(false);
  const [type, setType] = useState("meeting");
  const [summary, setSummary] = useState("");

  async function load() {
    const data = await apiFetch<ContactInteraction[]>(`/contacts/${contactId}/interactions/`);
    setInteractions(data);
  }

  useEffect(() => {
    load();
  }, [contactId]);

  async function handleAdd(e: React.FormEvent) {
    e.preventDefault();
    if (!summary.trim()) return;
    await apiFetch(`/contacts/${contactId}/interactions/`, {
      method: "POST",
      body: JSON.stringify({
        interaction_type: type,
        summary: summary.trim(),
        occurred_at: new Date().toISOString(),
      }),
    });
    setSummary("");
    setShowAdd(false);
    load();
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
          Interactions
        </h4>
        <Button size="sm" variant="ghost" className="h-6 text-xs" onClick={() => setShowAdd(!showAdd)}>
          <Plus className="h-3 w-3 mr-1" /> Log
        </Button>
      </div>

      {showAdd && (
        <form onSubmit={handleAdd} className="flex gap-1.5 mb-3">
          <Select value={type} onValueChange={setType}>
            <SelectTrigger className="h-7 text-xs w-28">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {INTERACTION_TYPES.map((t) => (
                <SelectItem key={t} value={t}>{t}</SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Input
            className="h-7 text-xs flex-1"
            placeholder="Summary..."
            value={summary}
            onChange={(e) => setSummary(e.target.value)}
          />
          <Button type="submit" size="sm" className="h-7 text-xs">Log</Button>
        </form>
      )}

      <div className="space-y-2">
        {interactions.map((interaction) => {
          const Icon = TYPE_ICONS[interaction.interaction_type] || MoreHorizontal;
          return (
            <div key={interaction.id} className="flex items-start gap-2 text-sm">
              <Icon className="h-3.5 w-3.5 text-muted-foreground mt-0.5 shrink-0" />
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-1.5">
                  <Badge variant="secondary" className="text-[10px]">{interaction.interaction_type}</Badge>
                  <span className="text-[10px] text-muted-foreground">
                    {formatDistanceToNow(new Date(interaction.occurred_at), { addSuffix: true })}
                  </span>
                </div>
                {interaction.summary && (
                  <p className="text-xs text-muted-foreground mt-0.5">{interaction.summary}</p>
                )}
              </div>
            </div>
          );
        })}
        {interactions.length === 0 && (
          <p className="text-xs text-muted-foreground text-center py-4">No interactions logged</p>
        )}
      </div>
    </div>
  );
}
