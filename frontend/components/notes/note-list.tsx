"use client";

import { useEffect, useState, useCallback } from "react";
import { apiFetch } from "@/lib/api";
import { Note } from "@/lib/types";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Plus, Pin } from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";
import { formatDistanceToNow } from "date-fns";

const NOTE_TYPES = [
  { value: "all", label: "All Types" },
  { value: "idea", label: "Idea" },
  { value: "meeting_note", label: "Meeting Note" },
  { value: "journal", label: "Journal" },
  { value: "research", label: "Research" },
  { value: "brainstorm", label: "Brainstorm" },
  { value: "decision_log", label: "Decision Log" },
  { value: "document", label: "Document" },
];

interface NoteListProps {
  selectedId: string | null;
  onSelect: (id: string) => void;
}

export function NoteList({ selectedId, onSelect }: NoteListProps) {
  const [notes, setNotes] = useState<Note[]>([]);
  const [loading, setLoading] = useState(true);
  const [filterType, setFilterType] = useState("all");
  const [showPinnedOnly, setShowPinnedOnly] = useState(false);

  const loadNotes = useCallback(async () => {
    const params = new URLSearchParams();
    if (filterType !== "all") params.set("note_type", filterType);
    if (showPinnedOnly) params.set("is_pinned", "true");
    const query = params.toString();
    const data = await apiFetch<Note[]>(`/notes/${query ? `?${query}` : ""}`);
    setNotes(data);
    setLoading(false);
  }, [filterType, showPinnedOnly]);

  useEffect(() => {
    loadNotes();
  }, [loadNotes]);

  async function handleNewNote() {
    const note = await apiFetch<Note>("/notes/", {
      method: "POST",
      body: JSON.stringify({ title: "Untitled note", content: "", note_type: "document" }),
    });
    await loadNotes();
    onSelect(note.id);
  }

  const pinned = notes.filter((n) => n.is_pinned);
  const unpinned = notes.filter((n) => !n.is_pinned);

  return (
    <div className="w-72 border-r border-border flex flex-col h-full">
      <div className="p-3 border-b border-border space-y-2">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold">Notes</h2>
          <Button size="sm" variant="ghost" onClick={handleNewNote}>
            <Plus className="h-4 w-4" />
          </Button>
        </div>
        <div className="flex gap-1">
          <Select value={filterType} onValueChange={setFilterType}>
            <SelectTrigger className="h-7 text-xs flex-1">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {NOTE_TYPES.map((t) => (
                <SelectItem key={t.value} value={t.value}>
                  {t.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button
            size="sm"
            variant={showPinnedOnly ? "default" : "ghost"}
            className="h-7 w-7 p-0"
            onClick={() => setShowPinnedOnly(!showPinnedOnly)}
          >
            <Pin className="h-3 w-3" />
          </Button>
        </div>
      </div>
      <ScrollArea className="flex-1">
        <div className="p-2 space-y-1">
          {loading && Array.from({ length: 5 }).map((_, i) => (
            <div key={i} className="px-2 py-2 space-y-2">
              <Skeleton className="h-4 w-3/4" />
              <Skeleton className="h-3 w-1/2" />
            </div>
          ))}
          {!loading && pinned.length > 0 && !showPinnedOnly && (
            <p className="text-[10px] text-muted-foreground px-2 pt-1 uppercase tracking-wider">Pinned</p>
          )}
          {pinned.map((note) => (
            <NoteItem key={note.id} note={note} selected={note.id === selectedId} onSelect={onSelect} />
          ))}
          {unpinned.length > 0 && pinned.length > 0 && !showPinnedOnly && (
            <p className="text-[10px] text-muted-foreground px-2 pt-2 uppercase tracking-wider">All Notes</p>
          )}
          {unpinned.map((note) => (
            <NoteItem key={note.id} note={note} selected={note.id === selectedId} onSelect={onSelect} />
          ))}
          {!loading && notes.length === 0 && (
            <p className="text-xs text-muted-foreground text-center py-8">No notes yet</p>
          )}
        </div>
      </ScrollArea>
    </div>
  );
}

function NoteItem({ note, selected, onSelect }: { note: Note; selected: boolean; onSelect: (id: string) => void }) {
  return (
    <button
      onClick={() => onSelect(note.id)}
      className={`w-full text-left rounded-lg px-2 py-2 text-sm transition-colors ${
        selected ? "bg-accent text-accent-foreground" : "hover:bg-accent/50"
      }`}
    >
      <div className="flex items-center gap-1.5">
        {note.is_pinned && <Pin className="h-3 w-3 text-muted-foreground shrink-0" />}
        <span className="truncate font-medium">{note.title || "Untitled"}</span>
      </div>
      <div className="flex items-center gap-1.5 mt-1">
        <Badge variant="secondary" className="text-[10px] px-1 py-0">
          {note.note_type}
        </Badge>
        <span className="text-[10px] text-muted-foreground">
          {formatDistanceToNow(new Date(note.updated_at), { addSuffix: true })}
        </span>
      </div>
    </button>
  );
}
