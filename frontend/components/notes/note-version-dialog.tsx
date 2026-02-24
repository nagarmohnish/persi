"use client";

import { useEffect, useState } from "react";
import { apiFetch } from "@/lib/api";
import { NoteVersion } from "@/lib/types";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { ScrollArea } from "@/components/ui/scroll-area";
import { formatDistanceToNow } from "date-fns";

interface NoteVersionDialogProps {
  noteId: string;
  open: boolean;
  onClose: () => void;
}

export function NoteVersionDialog({ noteId, open, onClose }: NoteVersionDialogProps) {
  const [versions, setVersions] = useState<NoteVersion[]>([]);
  const [selectedVersion, setSelectedVersion] = useState<NoteVersion | null>(null);

  useEffect(() => {
    if (open && noteId) {
      apiFetch<NoteVersion[]>(`/notes/${noteId}/versions`).then(setVersions);
    }
  }, [open, noteId]);

  return (
    <Dialog open={open} onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="max-w-2xl max-h-[70vh]">
        <DialogHeader>
          <DialogTitle>Version History</DialogTitle>
        </DialogHeader>
        {versions.length === 0 ? (
          <p className="text-sm text-muted-foreground py-4 text-center">No previous versions</p>
        ) : (
          <div className="flex gap-4 h-[400px]">
            <ScrollArea className="w-48 border-r border-border pr-2">
              <div className="space-y-1">
                {versions.map((v) => (
                  <button
                    key={v.id}
                    onClick={() => setSelectedVersion(v)}
                    className={`w-full text-left rounded px-2 py-1.5 text-xs transition-colors ${
                      selectedVersion?.id === v.id ? "bg-accent" : "hover:bg-accent/50"
                    }`}
                  >
                    <div className="font-medium">Version {v.version_number}</div>
                    <div className="text-muted-foreground">
                      {formatDistanceToNow(new Date(v.created_at), { addSuffix: true })}
                    </div>
                    {v.change_summary && (
                      <div className="text-muted-foreground truncate mt-0.5">{v.change_summary}</div>
                    )}
                  </button>
                ))}
              </div>
            </ScrollArea>
            <ScrollArea className="flex-1">
              {selectedVersion ? (
                <pre className="text-xs font-mono whitespace-pre-wrap text-foreground/80 p-2">
                  {selectedVersion.content || "(empty)"}
                </pre>
              ) : (
                <p className="text-sm text-muted-foreground text-center py-8">
                  Select a version to view
                </p>
              )}
            </ScrollArea>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
