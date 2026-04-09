import Image from "next/image";

export default function Home() {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-50 dark:bg-zinc-950 font-sans p-8">
      <main className="max-w-4xl w-full bg-white dark:bg-zinc-900 rounded-2xl shadow-xl overflow-hidden border border-gray-200 dark:border-zinc-800">
        <div className="p-8 sm:p-12">
          <div className="flex items-center gap-4 mb-8">
            <div className="w-12 h-12 bg-blue-600 rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-2xl">G</span>
            </div>
            <div>
              <h1 className="text-3xl font-bold text-gray-900 dark:text-white tracking-tight">GovOn</h1>
              <p className="text-gray-500 dark:text-zinc-400">On-Device AI Civil Complaint Analysis and Processing System</p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-12">
            <div className="p-6 bg-blue-50 dark:bg-blue-900/20 rounded-xl border border-blue-100 dark:border-blue-800/30">
              <h2 className="text-xl font-semibold text-blue-900 dark:text-blue-300 mb-3">Start Complaint Analysis</h2>
              <p className="text-blue-700 dark:text-blue-400/80 mb-4 text-sm">Analyze complaint content and generate draft responses using a local LLM.</p>
              <button className="px-6 py-2 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 transition-colors">New Analysis</button>
            </div>

            <div className="p-6 bg-zinc-50 dark:bg-zinc-800/50 rounded-xl border border-zinc-200 dark:border-zinc-700/50">
              <h2 className="text-xl font-semibold text-gray-900 dark:text-zinc-200 mb-3">Recent Activity</h2>
              <p className="text-gray-600 dark:text-zinc-400 mb-4 text-sm">Review analysis results and statistics from previously processed complaints.</p>
              <button className="px-6 py-2 border border-zinc-300 dark:border-zinc-600 rounded-lg font-medium hover:bg-zinc-100 dark:hover:bg-zinc-700 transition-colors">View History</button>
            </div>
          </div>

          <div className="mt-12 pt-8 border-t border-gray-100 dark:border-zinc-800 flex flex-col sm:flex-row justify-between items-center gap-4">
            <div className="flex gap-6 text-sm text-gray-500 dark:text-zinc-500 font-medium">
              <a href="#" className="hover:text-blue-600 transition-colors">System Settings</a>
              <a href="#" className="hover:text-blue-600 transition-colors">API Docs</a>
              <a href="#" className="hover:text-blue-600 transition-colors">Data Management</a>
            </div>
            <p className="text-xs text-gray-400 dark:text-zinc-600">© 2026 GovOn Project. Local-First AI System.</p>
          </div>
        </div>
      </main>
    </div>
  );
}
