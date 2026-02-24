"use client";

import { useEffect, useState } from "react";
import { apiFetch } from "@/lib/api";
import { MeetingParticipant } from "@/lib/types";
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
import { Plus, X, User } from "lucide-react";

const ROLES = ["organizer", "presenter", "attendee", "optional"];

interface ParticipantListProps {
  meetingId: string;
}

export function ParticipantList({ meetingId }: ParticipantListProps) {
  const [participants, setParticipants] = useState<MeetingParticipant[]>([]);
  const [showAdd, setShowAdd] = useState(false);
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [role, setRole] = useState("attendee");

  async function load() {
    const data = await apiFetch<MeetingParticipant[]>(`/meetings/${meetingId}/participants`);
    setParticipants(data);
  }

  useEffect(() => {
    load();
  }, [meetingId]);

  async function handleAdd(e: React.FormEvent) {
    e.preventDefault();
    if (!name.trim()) return;
    await apiFetch(`/meetings/${meetingId}/participants`, {
      method: "POST",
      body: JSON.stringify({ name: name.trim(), email: email.trim() || undefined, role }),
    });
    setName("");
    setEmail("");
    setRole("attendee");
    setShowAdd(false);
    load();
  }

  async function handleRemove(participantId: string) {
    await apiFetch(`/meetings/${meetingId}/participants/${participantId}`, { method: "DELETE" });
    load();
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
          Participants ({participants.length})
        </h4>
        <Button size="sm" variant="ghost" className="h-6 text-xs" onClick={() => setShowAdd(!showAdd)}>
          <Plus className="h-3 w-3 mr-1" /> Add
        </Button>
      </div>

      {showAdd && (
        <form onSubmit={handleAdd} className="flex gap-1.5 mb-3">
          <Input className="h-7 text-xs" placeholder="Name" value={name} onChange={(e) => setName(e.target.value)} />
          <Input className="h-7 text-xs" placeholder="Email" value={email} onChange={(e) => setEmail(e.target.value)} />
          <Select value={role} onValueChange={setRole}>
            <SelectTrigger className="h-7 text-xs w-28">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {ROLES.map((r) => (
                <SelectItem key={r} value={r}>{r}</SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button type="submit" size="sm" className="h-7 text-xs">Add</Button>
        </form>
      )}

      <div className="space-y-1">
        {participants.map((p) => (
          <div key={p.id} className="flex items-center gap-2 rounded px-2 py-1.5 text-sm hover:bg-accent/30">
            <User className="h-3.5 w-3.5 text-muted-foreground" />
            <span className="flex-1 truncate">{p.name}</span>
            {p.email && <span className="text-xs text-muted-foreground truncate">{p.email}</span>}
            <Badge variant="secondary" className="text-[10px]">{p.role}</Badge>
            <button onClick={() => handleRemove(p.id)} className="text-muted-foreground hover:text-destructive">
              <X className="h-3 w-3" />
            </button>
          </div>
        ))}
        {participants.length === 0 && (
          <p className="text-xs text-muted-foreground text-center py-2">No participants added</p>
        )}
      </div>
    </div>
  );
}
