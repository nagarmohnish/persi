"use client";

import { useEffect, useState, useCallback } from "react";
import { apiFetch } from "@/lib/api";
import { Trip } from "@/lib/types";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Plus, Plane } from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";
import { format } from "date-fns";

const STATUSES = [
  { value: "all", label: "All Status" },
  { value: "planning", label: "Planning" },
  { value: "booked", label: "Booked" },
  { value: "in_progress", label: "In Progress" },
  { value: "completed", label: "Completed" },
  { value: "cancelled", label: "Cancelled" },
];

const STATUS_COLORS: Record<string, string> = {
  planning: "bg-yellow-500/10 text-yellow-400 border-yellow-500/20",
  booked: "bg-blue-500/10 text-blue-400 border-blue-500/20",
  in_progress: "bg-green-500/10 text-green-400 border-green-500/20",
  completed: "bg-gray-500/10 text-gray-400 border-gray-500/20",
  cancelled: "bg-red-500/10 text-red-400 border-red-500/20",
};

interface TripListProps {
  selectedId: string | null;
  onSelect: (id: string) => void;
  onNewTrip: () => void;
  refreshKey: number;
}

export function TripList({ selectedId, onSelect, onNewTrip, refreshKey }: TripListProps) {
  const [trips, setTrips] = useState<Trip[]>([]);
  const [loading, setLoading] = useState(true);
  const [statusFilter, setStatusFilter] = useState("all");

  const loadTrips = useCallback(async () => {
    const params = new URLSearchParams();
    if (statusFilter !== "all") params.set("status", statusFilter);
    const query = params.toString();
    const data = await apiFetch<Trip[]>(`/travel/${query ? `?${query}` : ""}`);
    setTrips(data);
    setLoading(false);
  }, [statusFilter, refreshKey]);

  useEffect(() => {
    loadTrips();
  }, [loadTrips]);

  return (
    <div className="w-72 border-r border-border flex flex-col h-full">
      <div className="p-3 border-b border-border space-y-2">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold">Trips</h2>
          <Button size="sm" variant="ghost" onClick={onNewTrip}>
            <Plus className="h-4 w-4" />
          </Button>
        </div>
        <Select value={statusFilter} onValueChange={setStatusFilter}>
          <SelectTrigger className="h-7 text-xs">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {STATUSES.map((s) => (
              <SelectItem key={s.value} value={s.value}>{s.label}</SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      <ScrollArea className="flex-1">
        <div className="p-2 space-y-1">
          {loading && Array.from({ length: 3 }).map((_, i) => (
            <div key={i} className="px-3 py-2.5 space-y-2">
              <Skeleton className="h-4 w-3/4" />
              <Skeleton className="h-3 w-1/2" />
            </div>
          ))}
          {!loading && trips.map((trip) => (
            <button
              key={trip.id}
              onClick={() => onSelect(trip.id)}
              className={`w-full text-left rounded-lg px-3 py-2.5 text-sm transition-colors ${
                trip.id === selectedId ? "bg-accent text-accent-foreground" : "hover:bg-accent/50"
              }`}
            >
              <div className="flex items-center gap-2">
                <Plane className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
                <span className="truncate font-medium">{trip.title}</span>
              </div>
              <div className="flex items-center gap-1.5 mt-1 ml-5">
                <Badge variant="outline" className={`text-[10px] px-1 py-0 ${STATUS_COLORS[trip.status] || ""}`}>
                  {trip.status}
                </Badge>
                {trip.destination_city && (
                  <span className="text-[10px] text-muted-foreground truncate">{trip.destination_city}</span>
                )}
                {trip.start_date && (
                  <span className="text-[10px] text-muted-foreground">
                    {format(new Date(trip.start_date), "MMM d")}
                  </span>
                )}
              </div>
            </button>
          ))}
          {!loading && trips.length === 0 && (
            <p className="text-xs text-muted-foreground text-center py-8">No trips found</p>
          )}
        </div>
      </ScrollArea>
    </div>
  );
}
