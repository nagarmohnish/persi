"use client";

import { useState } from "react";
import { NoteList } from "@/components/notes/note-list";
import { NoteEditor } from "@/components/notes/note-editor";

export default function NotesPage() {
  const [selectedNoteId, setSelectedNoteId] = useState<string | null>(null);

  return (
    <div className="flex h-full">
      <NoteList selectedId={selectedNoteId} onSelect={setSelectedNoteId} />
      <NoteEditor
        noteId={selectedNoteId}
        onDeleted={() => setSelectedNoteId(null)}
      />
    </div>
  );
}
