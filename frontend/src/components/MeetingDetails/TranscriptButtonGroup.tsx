"use client";

import { useState, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { ButtonGroup } from '@/components/ui/button-group';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Copy, Download, FolderOpen, RefreshCw, Loader2, Users } from 'lucide-react';
import { invoke } from '@tauri-apps/api/core';
import Analytics from '@/lib/analytics';
import { RetranscribeDialog } from './RetranscribeDialog';
import { useConfig } from '@/contexts/ConfigContext';
import { toast } from 'sonner';
import { SpeakerIdentificationResult } from '@/types';


interface TranscriptButtonGroupProps {
  transcriptCount: number;
  onCopyTranscript: () => void;
  onOpenMeetingFolder: () => Promise<void>;
  meetingId?: string;
  meetingFolderPath?: string | null;
  onRefetchTranscripts?: () => Promise<void>;
  isRecording?: boolean;
}


export function TranscriptButtonGroup({
  transcriptCount,
  onCopyTranscript,
  onOpenMeetingFolder,
  meetingId,
  meetingFolderPath,
  onRefetchTranscripts,
  isRecording = false,
}: TranscriptButtonGroupProps) {
  const { betaFeatures } = useConfig();
  const [showRetranscribeDialog, setShowRetranscribeDialog] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [isIdentifyingSpeakers, setIsIdentifyingSpeakers] = useState(false);

  const handleExport = useCallback(async (format: string) => {
    if (!meetingId) return;
    setIsExporting(true);
    try {
      const result = await invoke<{ saved: boolean; path?: string; segment_count: number; skipped_count?: number }>('api_export_transcript', {
        meetingId,
        format,
      });
      if (result.saved) {
        toast.success(`Exported as ${format.toUpperCase()}`, {
          description: `${result.segment_count} segments saved${result.skipped_count ? ` (${result.skipped_count} skipped)` : ''}`,
          duration: 4000,
        });
      }
    } catch (error) {
      if (error instanceof Error && error.message !== 'Save cancelled') {
        toast.error('Export failed', {
          description: error instanceof Error ? error.message : String(error),
          duration: 5000,
        });
      }
    } finally {
      setIsExporting(false);
    }
  }, [meetingId]);

  const handleIdentifySpeakers = useCallback(async () => {
    if (!meetingId) return;

    setIsIdentifyingSpeakers(true);
    try {
      const result = await invoke<SpeakerIdentificationResult>('api_identify_meeting_speakers', {
        meetingId,
      });

      if (onRefetchTranscripts) {
        await onRefetchTranscripts();
      }

      toast.success('Speaker identification complete', {
        description: `${result.speaker_turn_count} speaker turns processed; ${result.updated_transcript_count} transcript segments labeled`,
        duration: 5000,
      });
    } catch (error) {
      toast.error('Speaker identification failed', {
        description: error instanceof Error ? error.message : String(error),
        duration: 7000,
      });
    } finally {
      setIsIdentifyingSpeakers(false);
    }
  }, [meetingId, onRefetchTranscripts]);

  const handleRetranscribeComplete = useCallback(async () => {
    // Refetch transcripts to show the updated data
    if (onRefetchTranscripts) {
      await onRefetchTranscripts();
    }
  }, [onRefetchTranscripts]);

  return (
    <div className="flex items-center justify-center w-full gap-2">
      <ButtonGroup>
        <Button
          variant="outline"
          size="sm"
          onClick={() => {
            Analytics.trackButtonClick('copy_transcript', 'meeting_details');
            onCopyTranscript();
          }}
          disabled={transcriptCount === 0}
          title={transcriptCount === 0 ? 'No transcript available' : 'Copy Transcript'}
        >
          <Copy />
          <span className="hidden lg:inline">Copy</span>
        </Button>

        <Button
          size="sm"
          variant="outline"
          className="xl:px-4"
          onClick={() => {
            Analytics.trackButtonClick('open_recording_folder', 'meeting_details');
            onOpenMeetingFolder();
          }}
          title="Open Recording Folder"
        >
          <FolderOpen className="xl:mr-2" size={18} />
          <span className="hidden lg:inline">Recording</span>
        </Button>

        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button
              size="sm"
              variant="outline"
              disabled={transcriptCount === 0 || isExporting}
              title={isExporting ? 'Exporting...' : transcriptCount === 0 ? 'No transcript to export' : 'Export Transcript'}
            >
              {isExporting ? <Loader2 className="animate-spin" size={18} /> : <Download size={18} />}
              <span>{isExporting ? 'Exporting...' : 'Export'}</span>
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuItem
              onClick={() => {
                Analytics.trackButtonClick('export_txt', 'meeting_details');
                handleExport('txt');
              }}
            >
              Export as TXT
            </DropdownMenuItem>
            <DropdownMenuItem
              onClick={() => {
                Analytics.trackButtonClick('export_vtt', 'meeting_details');
                handleExport('vtt');
              }}
            >
              Export as VTT
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>

        <Button
          size="sm"
          variant="outline"
          disabled={!meetingId || transcriptCount === 0 || isRecording || isIdentifyingSpeakers}
          onClick={() => {
            Analytics.trackButtonClick('identify_speakers', 'meeting_details');
            handleIdentifySpeakers();
          }}
          title={
            isRecording
              ? 'Speaker identification is available after recording'
              : transcriptCount === 0
                ? 'No transcript to identify'
                : 'Identify Speakers'
          }
        >
          {isIdentifyingSpeakers ? <Loader2 className="animate-spin" size={18} /> : <Users size={18} />}
          <span className="hidden xl:inline">{isIdentifyingSpeakers ? 'Identifying...' : 'Speakers'}</span>
        </Button>

        {betaFeatures.importAndRetranscribe && meetingId && meetingFolderPath && (
          <Button
            size="sm"
            variant="outline"
            className="bg-gradient-to-r from-blue-50 to-purple-50 hover:from-blue-100 hover:to-purple-100 border-blue-200 xl:px-4"
            onClick={() => {
              Analytics.trackButtonClick('enhance_transcript', 'meeting_details');
              setShowRetranscribeDialog(true);
            }}
            title="Retranscribe to enhance your recorded audio"
          >
            <RefreshCw className="xl:mr-2" size={18} />
            <span className="hidden lg:inline">Enhance</span>
          </Button>
        )}
      </ButtonGroup>

      {betaFeatures.importAndRetranscribe && meetingId && meetingFolderPath && (
        <RetranscribeDialog
          open={showRetranscribeDialog}
          onOpenChange={setShowRetranscribeDialog}
          meetingId={meetingId}
          meetingFolderPath={meetingFolderPath}
          onComplete={handleRetranscribeComplete}
        />
      )}
    </div>
  );
}
