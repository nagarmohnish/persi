"use client";

import { useState } from "react";
import { Trip } from "@/lib/types";
import { TripList } from "@/components/travel/trip-list";
import { TripDetail } from "@/components/travel/trip-detail";
import { TripFormDialog } from "@/components/travel/trip-form-dialog";

export default function TravelPage() {
  const [selectedTripId, setSelectedTripId] = useState<string | null>(null);
  const [showForm, setShowForm] = useState(false);
  const [editingTrip, setEditingTrip] = useState<Trip | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);

  function handleNewTrip() {
    setEditingTrip(null);
    setShowForm(true);
  }

  function handleEdit(trip: Trip) {
    setEditingTrip(trip);
    setShowForm(true);
  }

  function handleSaved() {
    setRefreshKey((k) => k + 1);
  }

  return (
    <div className="flex h-full">
      <TripList
        selectedId={selectedTripId}
        onSelect={setSelectedTripId}
        onNewTrip={handleNewTrip}
        refreshKey={refreshKey}
      />
      <TripDetail
        key={selectedTripId}
        tripId={selectedTripId}
        onEdit={handleEdit}
        onDeleted={() => { setSelectedTripId(null); setRefreshKey((k) => k + 1); }}
      />
      <TripFormDialog
        open={showForm}
        onClose={() => setShowForm(false)}
        trip={editingTrip}
        onSaved={handleSaved}
      />
    </div>
  );
}
