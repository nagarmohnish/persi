"use client";

import { useEffect, useState } from "react";
import { apiFetch } from "@/lib/api";
import { Contact } from "@/lib/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Clock, Check } from "lucide-react";
import { format, isPast } from "date-fns";

interface FollowUpRemindersProps {
  onContactClick?: (contactId: string) => void;
  limit?: number;
}

export function FollowUpReminders({ onContactClick, limit }: FollowUpRemindersProps) {
  const [contacts, setContacts] = useState<Contact[]>([]);

  async function load() {
    const data = await apiFetch<Contact[]>("/contacts/follow-ups/");
    setContacts(limit ? data.slice(0, limit) : data);
  }

  useEffect(() => {
    load();
  }, []);

  async function handleDismiss(contactId: string) {
    await apiFetch(`/contacts/${contactId}`, {
      method: "PATCH",
      body: JSON.stringify({ next_follow_up_at: null }),
    });
    load();
  }

  if (contacts.length === 0) {
    return <p className="text-xs text-muted-foreground text-center py-4">No follow-ups due</p>;
  }

  return (
    <div className="space-y-2">
      {contacts.map((c) => {
        const overdue = c.next_follow_up_at && isPast(new Date(c.next_follow_up_at));
        return (
          <div key={c.id} className="flex items-center gap-2 text-sm">
            <Clock className={`h-3.5 w-3.5 shrink-0 ${overdue ? "text-red-400" : "text-yellow-400"}`} />
            <button
              className="flex-1 text-left truncate hover:text-primary"
              onClick={() => onContactClick?.(c.id)}
            >
              {c.name}
            </button>
            {c.company && <span className="text-xs text-muted-foreground">{c.company}</span>}
            {c.next_follow_up_at && (
              <Badge variant={overdue ? "destructive" : "secondary"} className="text-[10px]">
                {format(new Date(c.next_follow_up_at), "MMM d")}
              </Badge>
            )}
            <Button size="sm" variant="ghost" className="h-5 w-5 p-0" onClick={() => handleDismiss(c.id)}>
              <Check className="h-3 w-3" />
            </Button>
          </div>
        );
      })}
    </div>
  );
}
