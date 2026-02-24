"use client";

import { useEffect, useState, useRef, useCallback } from "react";
import { apiFetch } from "@/lib/api";
import { Note } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { NoteMarkdownPreview } from "./note-markdown-preview";
import { NoteVersionDialog } from "./note-version-dialog";
import { useToast } from "@/hooks/use-toast";
import { Pin, PinOff, Eye, EyeOff, Sparkles, History, Trash2, StickyNote } from "lucide-react";

const NOTE_TYPES = [
  { value: "document", label: "Document" },
  { value: "idea", label: "Idea" },
  { value: "meeting_note", label: "Meeting Note" },
  { value: "journal", label: "Journal" },
  { value: "research", label: "Research" },
  { value: "brainstorm", label: "Brainstorm" },
  { value: "decision_log", label: "Decision Log" },
];

interface NoteEditorProps {
  noteId: string | null;
  onDeleted: () => void;
  onUpdated?: () => void;
}

export function NoteEditor({ noteId, onDeleted, onUpdated }: NoteEditorProps) {
  const { toast } = useToast();
  const [note, setNote] = useState<Note | null>(null);
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [noteType, setNoteType] = useState("document");
  const [preview, setPreview] = useState(false);
  const [showVersions, setShowVersions] = useState(false);
  const [saving, setSaving] = useState(false);
  const [aiLoading, setAiLoading] = useState(false);
  const saveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const isLoadingRef = useRef(false);

  useEffect(() => {
    if (!noteId) {
      setNote(null);
      return;
    }
    isLoadingRef.current = true;
    apiFetch<Note>(`/notes/${noteId}`).then((n) => {
      setNote(n);
      setTitle(n.title);
      setContent(n.content || "");
      setNoteType(n.note_type);
      isLoadingRef.current = false;
    });
  }, [noteId]);

  const saveNote = useCallback(
    async (newTitle: string, newContent: string, newType: string) => {
      if (!noteId || isLoadingRef.current) return;
      setSaving(true);
      try {
        const updated = await apiFetch<Note>(`/notes/${noteId}`, {
          method: "PUT",
          body: JSON.stringify({ title: newTitle, content: newContent, note_type: newType }),
        });
        setNote(updated);
        onUpdated?.();
      } catch (err) {
        toast({ title: "Error", description: err instanceof Error ? err.message : "Failed to save note", variant: "destructive" });
      } finally {
        setSaving(false);
      }
    },
    [noteId, onUpdated, toast]
  );

  function debouncedSave(newTitle: string, newContent: string, newType: string) {
    if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
    saveTimerRef.current = setTimeout(() => saveNote(newTitle, newContent, newType), 600);
  }

  function handleTitleChange(val: string) {
    setTitle(val);
    debouncedSave(val, content, noteType);
  }

  function handleContentChange(val: string) {
    setContent(val);
    debouncedSave(title, val, noteType);
  }

  function handleTypeChange(val: string) {
    setNoteType(val);
    saveNote(title, content, val);
  }

  async function handleTogglePin() {
    if (!noteId) return;
    const updated = await apiFetch<Note>(`/notes/${noteId}/pin`, { method: "POST" });
    setNote(updated);
    onUpdated?.();
  }

  async function handleDelete() {
    if (!noteId) return;
    try {
      await apiFetch(`/notes/${noteId}`, { method: "DELETE" });
      toast({ title: "Note deleted", description: "The note has been removed." });
      onDeleted();
    } catch (err) {
      toast({ title: "Error", description: err instanceof Error ? err.message : "Failed to delete note", variant: "destructive" });
    }
  }

  async function handleAiEnhance(action: string) {
    if (!noteId || !content) return;
    setAiLoading(true);
    try {
      const res = await apiFetch<{ response: string }>("/ai/ask", {
        method: "POST",
        body: JSON.stringify({
          query: `${action} the following note content. Return only the enhanced content, no explanations:\n\n${content}`,
        }),
      });
      setContent(res.response);
      debouncedSave(title, res.response, noteType);
      toast({ title: "AI enhanced", description: "Note content has been updated." });
    } catch (err) {
      toast({ title: "Error", description: err instanceof Error ? err.message : "AI enhancement failed", variant: "destructive" });
    } finally {
      setAiLoading(false);
    }
  }

  if (!noteId) {
    return (
      <div className="flex-1 flex items-center justify-center text-muted-foreground">
        <div className="text-center">
          <StickyNote className="h-12 w-12 opacity-20 mx-auto mb-3" />
          <p className="text-sm">Select a note or create a new one</p>
        </div>
      </div>
    );
  }

  if (!note) return null;

  return (
    <div className="flex-1 flex flex-col h-full min-w-0">
      {/* Toolbar */}
      <div className="flex items-center gap-2 px-4 py-2 border-b border-border">
        <Select value={noteType} onValueChange={handleTypeChange}>
          <SelectTrigger className="h-7 w-32 text-xs">
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

        <div className="flex-1" />

        {saving && <span className="text-[10px] text-muted-foreground">Saving...</span>}

        <Button size="sm" variant="ghost" className="h-7" onClick={handleTogglePin}>
          {note.is_pinned ? <PinOff className="h-3.5 w-3.5" /> : <Pin className="h-3.5 w-3.5" />}
        </Button>

        <Button size="sm" variant="ghost" className="h-7" onClick={() => setPreview(!preview)}>
          {preview ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />}
        </Button>

        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button size="sm" variant="ghost" className="h-7" disabled={aiLoading}>
              <Sparkles className="h-3.5 w-3.5" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuItem onClick={() => handleAiEnhance("Summarize")}>
              Summarize
            </DropdownMenuItem>
            <DropdownMenuItem onClick={() => handleAiEnhance("Expand and add more detail to")}>
              Expand
            </DropdownMenuItem>
            <DropdownMenuItem onClick={() => handleAiEnhance("Improve the writing and clarity of")}>
              Improve
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>

        <Button size="sm" variant="ghost" className="h-7" onClick={() => setShowVersions(true)}>
          <History className="h-3.5 w-3.5" />
        </Button>

        <Button size="sm" variant="ghost" className="h-7 text-destructive" onClick={handleDelete}>
          <Trash2 className="h-3.5 w-3.5" />
        </Button>
      </div>

      {/* Title */}
      <div className="px-4 pt-4">
        <input
          value={title}
          onChange={(e) => handleTitleChange(e.target.value)}
          className="w-full bg-transparent text-xl font-semibold outline-none placeholder:text-muted-foreground"
          placeholder="Note title..."
        />
      </div>

      {/* Content */}
      <div className="flex-1 px-4 py-3 overflow-auto">
        {preview ? (
          <NoteMarkdownPreview content={content} />
        ) : (
          <Textarea
            value={content}
            onChange={(e) => handleContentChange(e.target.value)}
            className="w-full h-full min-h-[200px] resize-none border-0 bg-transparent text-sm focus-visible:ring-0 p-0"
            placeholder="Start writing..."
          />
        )}
      </div>

      <NoteVersionDialog
        noteId={noteId}
        open={showVersions}
        onClose={() => setShowVersions(false)}
      />
    </div>
  );
}
