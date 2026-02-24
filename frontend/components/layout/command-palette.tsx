"use client";

import { useEffect, useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import { Command } from "cmdk";
import {
  LayoutDashboard,
  MessageSquare,
  Video,
  StickyNote,
  Calendar,
  Plane,
  Map,
  Users,
  Settings,
  Search,
  Plus,
  Sparkles,
  Loader2,
} from "lucide-react";
import { apiFetch } from "@/lib/api";

interface SearchResult {
  id: string;
  type: string;
  title: string;
  subtitle?: string;
}

const TYPE_ICONS: Record<string, React.ComponentType<{ className?: string }>> = {
  note: StickyNote,
  meeting: Video,
  contact: Users,
  trip: Plane,
};

const TYPE_ROUTES: Record<string, string> = {
  note: "/notes",
  meeting: "/meetings",
  contact: "/contacts",
  trip: "/travel",
};

export function CommandPalette() {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [searching, setSearching] = useState(false);
  const router = useRouter();

  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.key === "k" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        setOpen((prev) => !prev);
      }
    };
    document.addEventListener("keydown", down);
    return () => document.removeEventListener("keydown", down);
  }, []);

  const doSearch = useCallback(async (q: string) => {
    if (q.length < 2) {
      setSearchResults([]);
      return;
    }
    setSearching(true);
    try {
      const res = await apiFetch<{ results: SearchResult[] }>("/ai/search", {
        method: "POST",
        body: JSON.stringify({ query: q, limit: 8 }),
      });
      setSearchResults(res.results);
    } catch {
      setSearchResults([]);
    } finally {
      setSearching(false);
    }
  }, []);

  useEffect(() => {
    const timer = setTimeout(() => doSearch(query), 300);
    return () => clearTimeout(timer);
  }, [query, doSearch]);

  function runAction(action: () => void) {
    action();
    setOpen(false);
    setQuery("");
    setSearchResults([]);
  }

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50">
      <div
        className="absolute inset-0 bg-background/80 backdrop-blur-sm"
        onClick={() => { setOpen(false); setQuery(""); setSearchResults([]); }}
      />
      <div className="absolute left-1/2 top-[20%] -translate-x-1/2 w-full max-w-lg">
        <Command className="rounded-xl border border-border bg-card shadow-2xl overflow-hidden">
          <div className="flex items-center border-b border-border px-3">
            <Search className="h-4 w-4 shrink-0 text-muted-foreground mr-2" />
            <Command.Input
              placeholder="Search or type a command..."
              className="flex h-12 w-full bg-transparent text-sm outline-none placeholder:text-muted-foreground"
              autoFocus
              value={query}
              onValueChange={setQuery}
            />
            {searching && <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />}
          </div>
          <Command.List className="max-h-[300px] overflow-y-auto p-2">
            <Command.Empty className="py-6 text-center text-sm text-muted-foreground">
              No results found.
            </Command.Empty>

            {searchResults.length > 0 && (
              <Command.Group heading="Search Results" className="text-xs text-muted-foreground px-2 py-1.5">
                {searchResults.map((r) => {
                  const Icon = TYPE_ICONS[r.type] || Search;
                  const route = TYPE_ROUTES[r.type] || "/dashboard";
                  return (
                    <CommandItem
                      key={`${r.type}-${r.id}`}
                      icon={Icon}
                      label={r.title}
                      subtitle={r.subtitle || r.type}
                      onSelect={() => runAction(() => router.push(route))}
                    />
                  );
                })}
              </Command.Group>
            )}

            <Command.Group heading="Navigation" className="text-xs text-muted-foreground px-2 py-1.5">
              <CommandItem icon={LayoutDashboard} label="Dashboard" onSelect={() => runAction(() => router.push("/dashboard"))} />
              <CommandItem icon={MessageSquare} label="Chat" onSelect={() => runAction(() => router.push("/chat"))} />
              <CommandItem icon={Video} label="Meetings" onSelect={() => runAction(() => router.push("/meetings"))} />
              <CommandItem icon={StickyNote} label="Notes" onSelect={() => runAction(() => router.push("/notes"))} />
              <CommandItem icon={Calendar} label="Calendar" onSelect={() => runAction(() => router.push("/calendar"))} />
              <CommandItem icon={Plane} label="Travel" onSelect={() => runAction(() => router.push("/travel"))} />
              <CommandItem icon={Map} label="Journey" onSelect={() => runAction(() => router.push("/journey"))} />
              <CommandItem icon={Users} label="Contacts" onSelect={() => runAction(() => router.push("/contacts"))} />
              <CommandItem icon={Settings} label="Settings" onSelect={() => runAction(() => router.push("/settings"))} />
            </Command.Group>

            <Command.Separator className="h-px bg-border my-1" />

            <Command.Group heading="Quick Actions" className="text-xs text-muted-foreground px-2 py-1.5">
              <CommandItem icon={Plus} label="New Chat" onSelect={() => runAction(() => router.push("/chat"))} />
              <CommandItem icon={Sparkles} label="Ask Persi" onSelect={() => runAction(() => router.push("/chat"))} />
              <CommandItem icon={Plus} label="New Note" onSelect={() => runAction(() => router.push("/notes"))} />
              <CommandItem icon={Plus} label="New Meeting" onSelect={() => runAction(() => router.push("/meetings"))} />
              <CommandItem icon={Plus} label="New Trip" onSelect={() => runAction(() => router.push("/travel"))} />
            </Command.Group>
          </Command.List>
        </Command>
      </div>
    </div>
  );
}

function CommandItem({
  icon: Icon,
  label,
  subtitle,
  onSelect,
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  subtitle?: string;
  onSelect: () => void;
}) {
  return (
    <Command.Item
      onSelect={onSelect}
      className="flex items-center gap-2 rounded-lg px-2 py-2 text-sm cursor-pointer text-foreground aria-selected:bg-accent aria-selected:text-accent-foreground"
    >
      <Icon className="h-4 w-4 text-muted-foreground" />
      <span>{label}</span>
      {subtitle && (
        <span className="ml-auto text-xs text-muted-foreground">{subtitle}</span>
      )}
    </Command.Item>
  );
}
