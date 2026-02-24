import { redirect } from "next/navigation";

export default function Home() {
  // Phase 2: check auth status here, redirect to /login if unauthenticated
  redirect("/dashboard");
}
