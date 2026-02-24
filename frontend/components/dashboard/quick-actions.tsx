"use client";

import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { MessageSquare, Video, StickyNote, Sparkles } from "lucide-react";

export function QuickActions() {
  const router = useRouter();

  return (
    <div className="flex gap-2">
      <Button variant="outline" size="sm" onClick={() => router.push("/chat")}>
        <MessageSquare className="h-4 w-4 mr-1.5" /> New Chat
      </Button>
      <Button variant="outline" size="sm" onClick={() => router.push("/meetings")}>
        <Video className="h-4 w-4 mr-1.5" /> New Meeting
      </Button>
      <Button variant="outline" size="sm" onClick={() => router.push("/notes")}>
        <StickyNote className="h-4 w-4 mr-1.5" /> New Note
      </Button>
      <Button variant="outline" size="sm" onClick={() => router.push("/chat")}>
        <Sparkles className="h-4 w-4 mr-1.5" /> Ask Persi
      </Button>
    </div>
  );
}
