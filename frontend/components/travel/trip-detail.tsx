"use client";

import { useEffect, useState } from "react";
import { apiFetch } from "@/lib/api";
import { Trip } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { TripItinerary } from "./trip-itinerary";
import { Plane, Calendar, MapPin, DollarSign, Edit, Trash2 } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { format } from "date-fns";

const STATUS_COLORS: Record<string, string> = {
  planning: "bg-yellow-500/10 text-yellow-400 border-yellow-500/20",
  booked: "bg-blue-500/10 text-blue-400 border-blue-500/20",
  in_progress: "bg-green-500/10 text-green-400 border-green-500/20",
  completed: "bg-gray-500/10 text-gray-400 border-gray-500/20",
  cancelled: "bg-red-500/10 text-red-400 border-red-500/20",
};

interface TripDetailProps {
  tripId: string | null;
  onEdit: (trip: Trip) => void;
  onDeleted: () => void;
}

export function TripDetail({ tripId, onEdit, onDeleted }: TripDetailProps) {
  const { toast } = useToast();
  const [trip, setTrip] = useState<Trip | null>(null);

  useEffect(() => {
    if (!tripId) { setTrip(null); return; }
    apiFetch<Trip>(`/travel/${tripId}`).then(setTrip);
  }, [tripId]);

  async function handleDelete() {
    if (!tripId) return;
    try {
      await apiFetch(`/travel/${tripId}`, { method: "DELETE" });
      toast({ title: "Trip deleted", description: "The trip has been removed." });
      onDeleted();
    } catch (err) {
      toast({ title: "Error", description: err instanceof Error ? err.message : "Failed to delete trip", variant: "destructive" });
    }
  }

  if (!tripId || !trip) {
    return (
      <div className="flex-1 flex items-center justify-center text-muted-foreground">
        <div className="text-center">
          <Plane className="h-12 w-12 opacity-20 mx-auto mb-3" />
          <p className="text-sm">Select a trip to view details</p>
        </div>
      </div>
    );
  }

  const destination = [trip.destination_city, trip.destination_country].filter(Boolean).join(", ");

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="max-w-2xl mx-auto p-6 space-y-6">
        {/* Header */}
        <div className="flex items-start justify-between">
          <div>
            <h2 className="text-xl font-semibold">{trip.title}</h2>
            <div className="flex items-center gap-2 mt-2">
              <Badge variant="outline" className={STATUS_COLORS[trip.status] || ""}>
                {trip.status}
              </Badge>
              <Badge variant="secondary">{trip.purpose.replace(/_/g, " ")}</Badge>
            </div>
          </div>
          <div className="flex gap-1">
            <Button size="sm" variant="ghost" onClick={() => onEdit(trip)}>
              <Edit className="h-4 w-4" />
            </Button>
            <Button size="sm" variant="ghost" className="text-destructive" onClick={handleDelete}>
              <Trash2 className="h-4 w-4" />
            </Button>
          </div>
        </div>

        {/* Info */}
        <div className="grid grid-cols-2 gap-3 text-sm">
          {destination && (
            <div className="flex items-center gap-2 text-muted-foreground">
              <MapPin className="h-4 w-4" />
              {destination}
            </div>
          )}
          {trip.start_date && trip.end_date && (
            <div className="flex items-center gap-2 text-muted-foreground">
              <Calendar className="h-4 w-4" />
              {format(new Date(trip.start_date), "MMM d")} – {format(new Date(trip.end_date), "MMM d, yyyy")}
            </div>
          )}
          {trip.budget_cents && (
            <div className="flex items-center gap-2 text-muted-foreground">
              <DollarSign className="h-4 w-4" />
              {(trip.budget_cents / 100).toFixed(2)} {trip.currency}
            </div>
          )}
        </div>

        {/* Notes */}
        {trip.notes && (
          <div>
            <h3 className="text-sm font-semibold mb-1">Notes</h3>
            <p className="text-sm text-muted-foreground whitespace-pre-wrap">{trip.notes}</p>
          </div>
        )}

        {/* Itinerary */}
        <div className="border-t border-border pt-4">
          <TripItinerary tripId={trip.id} />
        </div>
      </div>
    </div>
  );
}
