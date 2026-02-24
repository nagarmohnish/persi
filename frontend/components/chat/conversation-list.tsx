"use client";

import { useState, useEffect } from "react";
import { apiFetch } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Plus, MessageSquare, Trash2 } from "lucide-react";
import { cn } from "@/lib/utils";

interface Conversation {
  id: string;
  title: string;
  updated_at: string;
}

interface ConversationListProps {
  selectedId: string | null;
  onSelect: (id: string) => void;
}

export function ConversationList({
  selectedId,
  onSelect,
}: ConversationListProps) {
  const [conversations, setConversations] = useState<Conversation[]>([]);

  useEffect(() => {
    loadConversations();
  }, []);

  async function loadConversations() {
    try {
      const data = await apiFetch<Conversation[]>("/conversations/");
      setConversations(data);
    } catch (err) {
      console.error("Failed to load conversations:", err);
    }
  }

  async function handleNew() {
    try {
      const conv = await apiFetch<Conversation>("/conversations/", {
        method: "POST",
        body: JSON.stringify({ title: "New conversation" }),
      });
      setConversations((prev) => [conv, ...prev]);
      onSelect(conv.id);
    } catch (err) {
      console.error("Failed to create conversation:", err);
    }
  }

  async function handleDelete(e: React.MouseEvent, id: string) {
    e.stopPropagation();
    try {
      await apiFetch(`/conversations/${id}`, { method: "DELETE" });
      setConversations((prev) => prev.filter((c) => c.id !== id));
      if (selectedId === id) {
        onSelect("");
      }
    } catch (err) {
      console.error("Failed to delete conversation:", err);
    }
  }

  return (
    <div className="w-72 border-r border-border flex flex-col bg-card/50">
      <div className="p-3 border-b border-border">
        <Button
          onClick={handleNew}
          className="w-full"
          variant="outline"
          size="sm"
        >
          <Plus className="h-4 w-4 mr-2" /> New Chat
        </Button>
      </div>
      <ScrollArea className="flex-1">
        <div className="p-2 space-y-0.5">
          {conversations.length === 0 && (
            <p className="text-xs text-muted-foreground text-center py-8">
              No conversations yet.
              <br />
              Start a new chat!
            </p>
          )}
          {conversations.map((conv) => (
            <button
              key={conv.id}
              onClick={() => onSelect(conv.id)}
              className={cn(
                "w-full text-left px-3 py-2 text-sm rounded-lg flex items-center gap-2 group transition-colors",
                selectedId === conv.id
                  ? "bg-accent text-accent-foreground"
                  : "text-muted-foreground hover:bg-accent/50 hover:text-foreground"
              )}
            >
              <MessageSquare className="h-4 w-4 shrink-0" />
              <span className="truncate flex-1">{conv.title}</span>
              <Trash2
                className="h-3.5 w-3.5 shrink-0 opacity-0 group-hover:opacity-60 hover:!opacity-100 transition-opacity"
                onClick={(e) => handleDelete(e, conv.id)}
              />
            </button>
          ))}
        </div>
      </ScrollArea>
    </div>
  );
}
