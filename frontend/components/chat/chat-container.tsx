"use client";

import { useState, useEffect, useRef } from "react";
import { MessageBubble } from "./message-bubble";
import { MessageInput } from "./message-input";
import { apiFetch } from "@/lib/api";
import { ScrollArea } from "@/components/ui/scroll-area";
import { MessageSquare } from "lucide-react";

interface Message {
  id: string;
  role: "user" | "assistant" | "system" | "tool";
  content: string;
  created_at: string;
}

interface ChatContainerProps {
  conversationId: string | null;
}

export function ChatContainer({ conversationId }: ChatContainerProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (conversationId) {
      loadMessages();
    } else {
      setMessages([]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [conversationId]);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function loadMessages() {
    try {
      const data = await apiFetch<Message[]>(
        `/conversations/${conversationId}/messages`
      );
      setMessages(data);
    } catch (err) {
      console.error("Failed to load messages:", err);
    }
  }

  async function handleSend(content: string) {
    if (!conversationId || !content.trim()) return;
    setLoading(true);

    // Optimistic update
    const tempId = `temp-${Date.now()}`;
    const tempMsg: Message = {
      id: tempId,
      role: "user",
      content,
      created_at: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, tempMsg]);

    try {
      // Save user message
      const savedMsg = await apiFetch<Message>(
        `/conversations/${conversationId}/messages`,
        {
          method: "POST",
          body: JSON.stringify({ role: "user", content }),
        }
      );

      // Replace temp with saved
      setMessages((prev) =>
        prev.map((m) => (m.id === tempId ? savedMsg : m))
      );

      // Generate AI response via Claude
      const assistantMsg = await apiFetch<Message>(
        `/conversations/${conversationId}/ai-reply`,
        { method: "POST" }
      );
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      console.error("Failed to send message:", err);
      // Remove optimistic message on error
      setMessages((prev) => prev.filter((m) => m.id !== tempId));
    } finally {
      setLoading(false);
    }
  }

  if (!conversationId) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center text-muted-foreground gap-3">
        <MessageSquare className="h-12 w-12 opacity-20" />
        <div className="text-center">
          <p className="text-lg font-medium">Welcome to Persi</p>
          <p className="text-sm mt-1">
            Select a conversation or start a new one
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-1 flex-col min-w-0">
      <ScrollArea className="flex-1">
        <div className="max-w-3xl mx-auto space-y-4 p-4 pb-2">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center py-20 text-muted-foreground">
              <p className="text-sm">
                Start the conversation. Ask anything about building your
                startup.
              </p>
            </div>
          )}
          {messages.map((msg) => (
            <MessageBubble key={msg.id} message={msg} />
          ))}
          <div ref={scrollRef} />
        </div>
      </ScrollArea>
      <div className="max-w-3xl mx-auto w-full">
        <MessageInput onSend={handleSend} disabled={loading} />
      </div>
    </div>
  );
}
