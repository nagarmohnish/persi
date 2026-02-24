"use client";

import { useEffect, useState } from "react";
import { apiFetch } from "@/lib/api";
import { TripItem } from "@/lib/types";
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
import { Plus, X, Plane, Hotel, Car, UtensilsCrossed, MapPin, Briefcase, Train, MoreHorizontal } from "lucide-react";
import { format } from "date-fns";

const ITEM_TYPES = ["flight", "hotel", "car_rental", "restaurant", "activity", "meeting", "transport", "other"];
const TYPE_ICONS: Record<string, typeof Plane> = {
  flight: Plane,
  hotel: Hotel,
  car_rental: Car,
  restaurant: UtensilsCrossed,
  activity: MapPin,
  meeting: Briefcase,
  transport: Train,
  other: MoreHorizontal,
};

interface TripItineraryProps {
  tripId: string;
}

export function TripItinerary({ tripId }: TripItineraryProps) {
  const [items, setItems] = useState<TripItem[]>([]);
  const [showAdd, setShowAdd] = useState(false);
  const [itemType, setItemType] = useState("other");
  const [itemTitle, setItemTitle] = useState("");
  const [itemLocation, setItemLocation] = useState("");
  const [itemStart, setItemStart] = useState("");

  async function load() {
    const data = await apiFetch<TripItem[]>(`/travel/${tripId}/items/`);
    setItems(data);
  }

  useEffect(() => {
    load();
  }, [tripId]);

  async function handleAdd(e: React.FormEvent) {
    e.preventDefault();
    if (!itemTitle.trim()) return;
    await apiFetch(`/travel/${tripId}/items/`, {
      method: "POST",
      body: JSON.stringify({
        item_type: itemType,
        title: itemTitle.trim(),
        location: itemLocation.trim() || undefined,
        start_time: itemStart ? new Date(itemStart).toISOString() : undefined,
        sort_order: items.length,
      }),
    });
    setItemTitle("");
    setItemLocation("");
    setItemStart("");
    setShowAdd(false);
    load();
  }

  async function handleRemove(itemId: string) {
    await apiFetch(`/travel/${tripId}/items/${itemId}`, { method: "DELETE" });
    load();
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold">Itinerary</h3>
        <Button size="sm" variant="outline" className="h-7 text-xs" onClick={() => setShowAdd(!showAdd)}>
          <Plus className="h-3 w-3 mr-1" /> Add Item
        </Button>
      </div>

      {showAdd && (
        <form onSubmit={handleAdd} className="border border-border rounded-lg p-3 mb-3 space-y-2">
          <div className="flex gap-2">
            <Select value={itemType} onValueChange={setItemType}>
              <SelectTrigger className="h-7 text-xs w-28"><SelectValue /></SelectTrigger>
              <SelectContent>
                {ITEM_TYPES.map((t) => (
                  <SelectItem key={t} value={t}>{t.replace(/_/g, " ")}</SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Input className="h-7 text-xs flex-1" placeholder="Title" value={itemTitle} onChange={(e) => setItemTitle(e.target.value)} />
          </div>
          <div className="flex gap-2">
            <Input className="h-7 text-xs flex-1" placeholder="Location" value={itemLocation} onChange={(e) => setItemLocation(e.target.value)} />
            <Input className="h-7 text-xs" type="datetime-local" value={itemStart} onChange={(e) => setItemStart(e.target.value)} />
            <Button type="submit" size="sm" className="h-7 text-xs">Add</Button>
          </div>
        </form>
      )}

      <div className="space-y-2">
        {items.map((item) => {
          const Icon = TYPE_ICONS[item.item_type] || MoreHorizontal;
          return (
            <div key={item.id} className="flex items-start gap-3 rounded-lg border border-border p-3">
              <Icon className="h-4 w-4 text-muted-foreground mt-0.5 shrink-0" />
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-medium">{item.title}</span>
                  <Badge variant="secondary" className="text-[10px]">{item.item_type.replace(/_/g, " ")}</Badge>
                  {item.status !== "researching" && (
                    <Badge variant="outline" className="text-[10px]">{item.status}</Badge>
                  )}
                </div>
                <div className="flex items-center gap-3 text-xs text-muted-foreground mt-1">
                  {item.start_time && <span>{format(new Date(item.start_time), "MMM d, h:mm a")}</span>}
                  {item.location && <span>{item.location}</span>}
                  {item.cost_cents && <span>${(item.cost_cents / 100).toFixed(2)} {item.currency}</span>}
                  {item.confirmation_number && <span>Conf: {item.confirmation_number}</span>}
                </div>
              </div>
              <button onClick={() => handleRemove(item.id)} className="text-muted-foreground hover:text-destructive">
                <X className="h-3.5 w-3.5" />
              </button>
            </div>
          );
        })}
        {items.length === 0 && !showAdd && (
          <p className="text-xs text-muted-foreground text-center py-4">No items in itinerary</p>
        )}
      </div>
    </div>
  );
}
