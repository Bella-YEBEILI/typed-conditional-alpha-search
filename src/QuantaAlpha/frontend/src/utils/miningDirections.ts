export interface MiningDirectionItem {
  id: string;
  label: string;
  description: string;
}

export const MINING_DIRECTIONS: MiningDirectionItem[] = [
  {
    id: 'pv',
    label: '价量因子 (PV)',
    description: '基于日频价格和成交量数据挖掘因子',
  },
  {
    id: 'minutes',
    label: '分钟因子 (Minutes)',
    description: '基于分钟级别高频数据挖掘因子',
  },
  {
    id: 'joint',
    label: '联合因子 (Joint)',
    description: '联合使用价量+分钟数据域挖掘因子',
  },
];

export function getDirectionLabel(id: string): string {
  const item = MINING_DIRECTIONS.find((d) => d.id === id);
  return item?.label || id;
}

export function getDefaultMiningDirection(): string {
  try {
    const raw = localStorage.getItem('quantaalpha_config');
    if (!raw) return 'pv';
    const config = JSON.parse(raw);
    return config?.defaultDirection || 'pv';
  } catch {
    return 'pv';
  }
}
