"use client";

import { useEffect, useState } from "react";
import { apiFetch } from "@/lib/api";
import { Contact } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { InteractionLog } from "./interaction-log";
import { Users, Mail, Phone, Linkedin, Twitter, Edit, Trash2, Calendar } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { format } from "date-fns";

const STRENGTH_COLORS: Record<string, string> = {
  cold: "bg-blue-500/10 text-blue-400 border-blue-500/20",
  warm: "bg-yellow-500/10 text-yellow-400 border-yellow-500/20",
  hot: "bg-orange-500/10 text-orange-400 border-orange-500/20",
  close: "bg-green-500/10 text-green-400 border-green-500/20",
};

interface ContactDetailProps {
  contactId: string | null;
  onEdit: (contact: Contact) => void;
  onDeleted: () => void;
}

export function ContactDetail({ contactId, onEdit, onDeleted }: ContactDetailProps) {
  const { toast } = useToast();
  const [contact, setContact] = useState<Contact | null>(null);
  const [followUpDate, setFollowUpDate] = useState("");

  useEffect(() => {
    if (!contactId) { setContact(null); return; }
    apiFetch<Contact>(`/contacts/${contactId}`).then((c) => {
      setContact(c);
      setFollowUpDate(c.next_follow_up_at?.split("T")[0] || "");
    });
  }, [contactId]);

  async function handleDelete() {
    if (!contactId) return;
    try {
      await apiFetch(`/contacts/${contactId}`, { method: "DELETE" });
      toast({ title: "Contact deleted", description: "The contact has been removed." });
      onDeleted();
    } catch (err) {
      toast({ title: "Error", description: err instanceof Error ? err.message : "Failed to delete contact", variant: "destructive" });
    }
  }

  async function handleFollowUpChange(date: string) {
    if (!contactId) return;
    setFollowUpDate(date);
    await apiFetch(`/contacts/${contactId}`, {
      method: "PATCH",
      body: JSON.stringify({ next_follow_up_at: date ? new Date(date).toISOString() : null }),
    });
  }

  if (!contactId || !contact) {
    return (
      <div className="flex-1 flex items-center justify-center text-muted-foreground">
        <div className="text-center">
          <Users className="h-12 w-12 opacity-20 mx-auto mb-3" />
          <p className="text-sm">Select a contact to view details</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="max-w-2xl mx-auto p-6 space-y-6">
        {/* Header */}
        <div className="flex items-start justify-between">
          <div>
            <h2 className="text-xl font-semibold">{contact.name}</h2>
            <div className="flex items-center gap-2 mt-1">
              {contact.title && <span className="text-sm text-muted-foreground">{contact.title}</span>}
              {contact.title && contact.company && <span className="text-muted-foreground">at</span>}
              {contact.company && <span className="text-sm text-muted-foreground font-medium">{contact.company}</span>}
            </div>
            <div className="flex items-center gap-2 mt-2">
              <Badge variant="outline" className={STRENGTH_COLORS[contact.relationship_strength] || ""}>
                {contact.relationship_strength}
              </Badge>
              <Badge variant="secondary">{contact.contact_type.replace(/_/g, " ")}</Badge>
              {contact.tags?.map((tag) => (
                <Badge key={tag} variant="outline" className="text-[10px]">{tag}</Badge>
              ))}
            </div>
          </div>
          <div className="flex gap-1">
            <Button size="sm" variant="ghost" onClick={() => onEdit(contact)}>
              <Edit className="h-4 w-4" />
            </Button>
            <Button size="sm" variant="ghost" className="text-destructive" onClick={handleDelete}>
              <Trash2 className="h-4 w-4" />
            </Button>
          </div>
        </div>

        {/* Contact Info */}
        <div className="grid grid-cols-2 gap-3 text-sm">
          {contact.email && (
            <div className="flex items-center gap-2 text-muted-foreground">
              <Mail className="h-4 w-4" />
              <a href={`mailto:${contact.email}`} className="text-primary hover:underline">{contact.email}</a>
            </div>
          )}
          {contact.phone && (
            <div className="flex items-center gap-2 text-muted-foreground">
              <Phone className="h-4 w-4" />
              {contact.phone}
            </div>
          )}
          {contact.linkedin_url && (
            <div className="flex items-center gap-2 text-muted-foreground">
              <Linkedin className="h-4 w-4" />
              <a href={contact.linkedin_url} target="_blank" rel="noopener noreferrer" className="text-primary hover:underline truncate">LinkedIn</a>
            </div>
          )}
          {contact.twitter_url && (
            <div className="flex items-center gap-2 text-muted-foreground">
              <Twitter className="h-4 w-4" />
              <a href={contact.twitter_url} target="_blank" rel="noopener noreferrer" className="text-primary hover:underline truncate">Twitter</a>
            </div>
          )}
        </div>

        {/* Follow-up */}
        <div className="flex items-center gap-3 text-sm">
          <Calendar className="h-4 w-4 text-muted-foreground" />
          <Label className="text-xs">Follow-up:</Label>
          <Input
            type="date"
            className="h-7 w-40 text-xs"
            value={followUpDate}
            onChange={(e) => handleFollowUpChange(e.target.value)}
          />
          {contact.last_contacted_at && (
            <span className="text-xs text-muted-foreground">
              Last contacted: {format(new Date(contact.last_contacted_at), "MMM d, yyyy")}
            </span>
          )}
        </div>

        {/* Notes */}
        {contact.notes && (
          <div>
            <h3 className="text-sm font-semibold mb-1">Notes</h3>
            <p className="text-sm text-muted-foreground whitespace-pre-wrap">{contact.notes}</p>
          </div>
        )}

        {/* Tabs */}
        <Tabs defaultValue="interactions">
          <TabsList>
            <TabsTrigger value="interactions">Interactions</TabsTrigger>
          </TabsList>
          <TabsContent value="interactions" className="mt-4">
            <InteractionLog contactId={contact.id} />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
