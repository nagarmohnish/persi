"use client";

import { useState } from "react";
import { ConversationList } from "@/components/chat/conversation-list";
import { ChatContainer } from "@/components/chat/chat-container";

export default function ChatPage() {
  const [selectedConversationId, setSelectedConversationId] = useState<
    string | null
  >(null);

  return (
    <div className="flex h-full">
      <ConversationList
        selectedId={selectedConversationId}
        onSelect={(id) => setSelectedConversationId(id || null)}
      />
      <ChatContainer conversationId={selectedConversationId} />
    </div>
  );
}
