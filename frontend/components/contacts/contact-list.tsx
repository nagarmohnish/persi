"use client";

import { useEffect, useState, useCallback } from "react";
import { apiFetch } from "@/lib/api";
import { Contact } from "@/lib/types";
import { ScrollArea } from "@/components/ui/scroll-area";
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
import { Plus, Search } from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";

const CONTACT_TYPES = [
  { value: "all", label: "All Types" },
  { value: "investor", label: "Investor" },
  { value: "advisor", label: "Advisor" },
  { value: "customer", label: "Customer" },
  { value: "partner", label: "Partner" },
  { value: "team_member", label: "Team" },
  { value: "mentor", label: "Mentor" },
  { value: "service_provider", label: "Service" },
  { value: "other", label: "Other" },
];

const STRENGTH_DOTS: Record<string, string> = {
  cold: "bg-blue-400",
  warm: "bg-yellow-400",
  hot: "bg-orange-400",
  close: "bg-green-400",
};

interface ContactListProps {
  selectedId: string | null;
  onSelect: (id: string) => void;
  onNewContact: () => void;
  refreshKey: number;
}

export function ContactList({ selectedId, onSelect, onNewContact, refreshKey }: ContactListProps) {
  const [contacts, setContacts] = useState<Contact[]>([]);
  const [loading, setLoading] = useState(true);
  const [typeFilter, setTypeFilter] = useState("all");
  const [searchQuery, setSearchQuery] = useState("");

  const loadContacts = useCallback(async () => {
    const params = new URLSearchParams();
    if (typeFilter !== "all") params.set("contact_type", typeFilter);
    const query = params.toString();
    const data = await apiFetch<Contact[]>(`/contacts/${query ? `?${query}` : ""}`);
    setContacts(data);
    setLoading(false);
  }, [typeFilter, refreshKey]);

  useEffect(() => {
    loadContacts();
  }, [loadContacts]);

  const filtered = contacts.filter((c) => {
    if (!searchQuery) return true;
    const q = searchQuery.toLowerCase();
    return (
      c.name.toLowerCase().includes(q) ||
      (c.email?.toLowerCase().includes(q)) ||
      (c.company?.toLowerCase().includes(q))
    );
  });

  return (
    <div className="w-80 border-r border-border flex flex-col h-full">
      <div className="p-3 border-b border-border space-y-2">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold">Contacts</h2>
          <Button size="sm" variant="ghost" onClick={onNewContact}>
            <Plus className="h-4 w-4" />
          </Button>
        </div>
        <div className="relative">
          <Search className="absolute left-2 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
          <Input
            className="h-7 text-xs pl-7"
            placeholder="Search contacts..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>
        <Select value={typeFilter} onValueChange={setTypeFilter}>
          <SelectTrigger className="h-7 text-xs">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {CONTACT_TYPES.map((t) => (
              <SelectItem key={t.value} value={t.value}>{t.label}</SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      <ScrollArea className="flex-1">
        <div className="p-2 space-y-1">
          {loading && Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="px-3 py-2.5 space-y-2">
              <Skeleton className="h-4 w-2/3" />
              <Skeleton className="h-3 w-1/2" />
            </div>
          ))}
          {!loading && filtered.map((c) => (
            <button
              key={c.id}
              onClick={() => onSelect(c.id)}
              className={`w-full text-left rounded-lg px-3 py-2.5 text-sm transition-colors ${
                c.id === selectedId ? "bg-accent text-accent-foreground" : "hover:bg-accent/50"
              }`}
            >
              <div className="flex items-center gap-2">
                <span className={`w-2 h-2 rounded-full shrink-0 ${STRENGTH_DOTS[c.relationship_strength] || "bg-muted"}`} />
                <span className="truncate font-medium">{c.name}</span>
              </div>
              <div className="flex items-center gap-1.5 mt-1 ml-4">
                {c.company && <span className="text-xs text-muted-foreground truncate">{c.company}</span>}
                <Badge variant="secondary" className="text-[10px] px-1 py-0">{c.contact_type}</Badge>
              </div>
            </button>
          ))}
          {!loading && filtered.length === 0 && (
            <p className="text-xs text-muted-foreground text-center py-8">No contacts found</p>
          )}
        </div>
      </ScrollArea>
    </div>
  );
}
