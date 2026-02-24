"use client";

import { useState, useEffect } from "react";
import { apiFetch } from "@/lib/api";
import { Trip } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useToast } from "@/hooks/use-toast";

const PURPOSES = ["investor_meeting", "conference", "team_offsite", "customer_visit", "personal", "other"];

interface TripFormDialogProps {
  open: boolean;
  onClose: () => void;
  trip?: Trip | null;
  onSaved: () => void;
}

export function TripFormDialog({ open, onClose, trip, onSaved }: TripFormDialogProps) {
  const { toast } = useToast();
  const [title, setTitle] = useState("");
  const [purpose, setPurpose] = useState("other");
  const [destinationCity, setDestinationCity] = useState("");
  const [destinationCountry, setDestinationCountry] = useState("");
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [budget, setBudget] = useState("");
  const [currency, setCurrency] = useState("USD");
  const [notes, setNotes] = useState("");
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    if (trip) {
      setTitle(trip.title);
      setPurpose(trip.purpose);
      setDestinationCity(trip.destination_city || "");
      setDestinationCountry(trip.destination_country || "");
      setStartDate(trip.start_date || "");
      setEndDate(trip.end_date || "");
      setBudget(trip.budget_cents ? String(trip.budget_cents / 100) : "");
      setCurrency(trip.currency);
      setNotes(trip.notes || "");
    } else {
      setTitle(""); setPurpose("other"); setDestinationCity(""); setDestinationCountry("");
      setStartDate(""); setEndDate(""); setBudget(""); setCurrency("USD"); setNotes("");
    }
  }, [trip, open]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!title.trim()) return;
    setSubmitting(true);
    try {
      const body = {
        title: title.trim(),
        purpose,
        destination_city: destinationCity.trim() || undefined,
        destination_country: destinationCountry.trim() || undefined,
        start_date: startDate || undefined,
        end_date: endDate || undefined,
        budget_cents: budget ? Math.round(Number(budget) * 100) : undefined,
        currency,
        notes: notes.trim() || undefined,
      };
      if (trip) {
        await apiFetch(`/travel/${trip.id}`, { method: "PATCH", body: JSON.stringify(body) });
        toast({ title: "Trip updated", description: "Your changes have been saved." });
      } else {
        await apiFetch("/travel/", { method: "POST", body: JSON.stringify(body) });
        toast({ title: "Trip created", description: "New trip has been added." });
      }
      onSaved();
      onClose();
    } catch (err) {
      toast({ title: "Error", description: err instanceof Error ? err.message : "Something went wrong", variant: "destructive" });
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <Dialog open={open} onOpenChange={(o) => !o && onClose()}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{trip ? "Edit Trip" : "New Trip"}</DialogTitle>
        </DialogHeader>
        <form onSubmit={handleSubmit} className="space-y-3">
          <div>
            <Label>Title</Label>
            <Input value={title} onChange={(e) => setTitle(e.target.value)} placeholder="e.g., SF Investor Meetings" />
          </div>
          <div className="flex gap-3">
            <div className="flex-1">
              <Label>Purpose</Label>
              <Select value={purpose} onValueChange={setPurpose}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  {PURPOSES.map((p) => (
                    <SelectItem key={p} value={p}>{p.replace(/_/g, " ")}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
          <div className="flex gap-3">
            <div className="flex-1">
              <Label>City</Label>
              <Input value={destinationCity} onChange={(e) => setDestinationCity(e.target.value)} placeholder="San Francisco" />
            </div>
            <div className="flex-1">
              <Label>Country</Label>
              <Input value={destinationCountry} onChange={(e) => setDestinationCountry(e.target.value)} placeholder="USA" />
            </div>
          </div>
          <div className="flex gap-3">
            <div className="flex-1">
              <Label>Start Date</Label>
              <Input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} />
            </div>
            <div className="flex-1">
              <Label>End Date</Label>
              <Input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} />
            </div>
          </div>
          <div className="flex gap-3">
            <div className="flex-1">
              <Label>Budget</Label>
              <Input type="number" step="0.01" value={budget} onChange={(e) => setBudget(e.target.value)} placeholder="0.00" />
            </div>
            <div className="w-24">
              <Label>Currency</Label>
              <Input value={currency} onChange={(e) => setCurrency(e.target.value)} />
            </div>
          </div>
          <div>
            <Label>Notes</Label>
            <Textarea value={notes} onChange={(e) => setNotes(e.target.value)} rows={2} />
          </div>
          <div className="flex justify-end gap-2">
            <Button type="button" variant="ghost" onClick={onClose}>Cancel</Button>
            <Button type="submit" disabled={submitting || !title.trim()}>
              {submitting ? "Saving..." : trip ? "Update" : "Create"}
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
}
