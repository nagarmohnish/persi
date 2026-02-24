"use client";

import { useState } from "react";
import { Contact } from "@/lib/types";
import { ContactList } from "@/components/contacts/contact-list";
import { ContactDetail } from "@/components/contacts/contact-detail";
import { ContactFormDialog } from "@/components/contacts/contact-form-dialog";

export default function ContactsPage() {
  const [selectedContactId, setSelectedContactId] = useState<string | null>(null);
  const [showForm, setShowForm] = useState(false);
  const [editingContact, setEditingContact] = useState<Contact | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);

  function handleNewContact() {
    setEditingContact(null);
    setShowForm(true);
  }

  function handleEdit(contact: Contact) {
    setEditingContact(contact);
    setShowForm(true);
  }

  function handleSaved() {
    setRefreshKey((k) => k + 1);
  }

  return (
    <div className="flex h-full">
      <ContactList
        selectedId={selectedContactId}
        onSelect={setSelectedContactId}
        onNewContact={handleNewContact}
        refreshKey={refreshKey}
      />
      <ContactDetail
        key={selectedContactId}
        contactId={selectedContactId}
        onEdit={handleEdit}
        onDeleted={() => { setSelectedContactId(null); setRefreshKey((k) => k + 1); }}
      />
      <ContactFormDialog
        open={showForm}
        onClose={() => setShowForm(false)}
        contact={editingContact}
        onSaved={handleSaved}
      />
    </div>
  );
}
