{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "%matplotlib inline\n",
    "from matplotlib import style\n",
    "style.use('fivethirtyeight')\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import datetime as dt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sqlalchemy\n",
    "from sqlalchemy.ext.automap import automap_base\n",
    "from sqlalchemy.orm import Session\n",
    "from sqlalchemy import create_engine, func, inspect"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "engine = create_engine(\"sqlite:///Resources/hawaii.sqlite\")\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['measurement', 'station']"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "inspector = inspect(engine)\n",
    "inspector.get_table_names()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['measurement', 'station']"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Base = automap_base()\n",
    "Base.prepare(engine, reflect=True)\n",
    "Base.classes.keys()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "Measurment = Base.classes.measurement\n",
    "Station = Base.classes.station\n",
    "session = Session(engine)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "id INTEGER\n",
      "station TEXT\n",
      "name TEXT\n",
      "latitude FLOAT\n",
      "longitude FLOAT\n",
      "elevation FLOAT\n"
     ]
    }
   ],
   "source": [
    "columns_stations = inspector.get_columns('station')\n",
    "for y in columns_stations:\n",
    "    print(y['name'], y[\"type\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "id INTEGER\n",
      "station TEXT\n",
      "date TEXT\n",
      "prcp FLOAT\n",
      "tobs FLOAT\n"
     ]
    }
   ],
   "source": [
    "columns = inspector.get_columns('measurement')\n",
    "for x in columns:\n",
    "    print(x['name'], x[\"type\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "sel = [Measurment.date, \n",
    "       Measurment.prcp]\n",
    "\n",
    "prcp_date = session.query(*sel).all()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "19550"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# last entry in list\n",
    "len(prcp_date)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Precipitation</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Date</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2016-08-07</th>\n",
       "      <td>1.30</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2016-08-08</th>\n",
       "      <td>0.02</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2016-08-09</th>\n",
       "      <td>0.56</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2016-08-10</th>\n",
       "      <td>0.00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2016-08-11</th>\n",
       "      <td>0.04</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2016-08-12</th>\n",
       "      <td>0.39</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2016-08-13</th>\n",
       "      <td>0.45</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2016-08-14</th>\n",
       "      <td>0.75</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2016-08-15</th>\n",
       "      <td>0.95</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2016-08-16</th>\n",
       "      <td>0.85</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "            Precipitation\n",
       "Date                     \n",
       "2016-08-07           1.30\n",
       "2016-08-08           0.02\n",
       "2016-08-09           0.56\n",
       "2016-08-10           0.00\n",
       "2016-08-11           0.04\n",
       "2016-08-12           0.39\n",
       "2016-08-13           0.45\n",
       "2016-08-14           0.75\n",
       "2016-08-15           0.95\n",
       "2016-08-16           0.85"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# create data frame with filters\n",
    "import datetime as dt\n",
    "import numpy as np\n",
    "\n",
    "\n",
    "weather_df = pd.DataFrame(prcp_date[19185:19550], columns=['Date', 'Precipitation'])\n",
    "weather_df.set_index('Date', inplace=True, )\n",
    "weather_df.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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YtSoUbh37x6GDRsmeRlERFQ1SN46olq1aqhWreTZlp6ejurVq0u+4mvXruG1115DZmYmateujVatWmHfvn3w8fGRvAwiIqoaJA+roKAg7N69G6+//nqR0x48eICNGzciODhY8hXHxsZKPi8REVVtkl8GnDhxIo4cOYI333wTZ8+eBQDcuHED+/fvx/PPP48rV67g3XfftVooERFVXZLXrDp37oxly5bhvffew7fffgsAiIiIgMFgQI0aNRATE4PWrVtbLZSIiKousz7RO2jQIPTq1QsHDx7E5cuXodfr8dRTTyEsLAzOzs7WaiQioirO7N1PODo6ok+fPtZoISIiKpbZw+ru3btITU3FnTt3TD4YXKB9+/YWCSMiIiogeVjpdDpERkZi69atyM/PBwAYDAbjZ64K/n/79m3rlBIRUZUleVhNmDABO3bswNixY9G+fXuTbw0mIiKyJsnDav/+/Xj99dcxZ84ca/YQEREVIflzVnZ2dvDz87NmCxERUbEkD6t+/fph37591mwhIiIqVokvA968edPk8FtvvYUxY8Zg3LhxGDNmDLy9vWFjY1Pkch4eHpavJCKiKq3EYdWwYcMie1c3GAxISkpCXFxciQvk1oBERGRpJQ6ryMjIYr8KhIiISG4lDquoqCg5O4iIiEokeQMLIiIipUgeVlFRUWjRokWJp7ds2RIffPCBRaKIiIgKkzys9u7di4EDB5Z4+oABA7B7926LRBERERUmeVilpaWV+pXzPj4+SEtLs0gUERFRYZKHlYuLC1JSUko8/cqVK3BwcLBEExERkQnJw6pTp06IjY0tdmClpKRg1apV6NSpkyXbiIiIAJixI9upU6di3759aN++PV566SUEBARAo9Hgt99+w7p162BjY4Np06ZZs5WIiKooycPKz88Pe/bswaRJkxATE2NyWvv27TF//nz4+/tbPJCIiMisbwpu3Lgxdu7ciczMTKSkpMBgMKBBgwaoWbNmhUM+/vhjREdHY+zYsViwYEGFl0dERJWH2V9rDwC1atVCrVq1LBaRkJCA1atXo0mTJhZbJhERVR5mDav8/HwcPHgQKSkpuHPnDgwGg8npGo0GkZGRZgVkZWVh7Nix+PLLLzF//nyzLktERFWD5GF15swZjBgxAqmpqUWGVIHyDKsJEyagX79+CA0N5bAiIqJiSR5WkyZNQk5ODtasWYP27dvDzc2twle+evVqXL58GcuWLZN8meTk5Apfr2U5Sm5SX7s07JafqO3slldFu1vHOyKhwz0L1ZinoF3qhnlmrVlFRUWhd+/e5Sv7h+TkZMyePRu7du2CnZ2d5MupbovD+DRJTcnJyeprl4Dd8hO1nd3yski3xOcvSytPu+RhVadOHdjalmt7jGKdOHECmZmZaNu2rfG4/Px8/PTTT4iNjcW1a9dgb29vsesjIiJxSZ4+4eHhWL9+PcLDw/HEE09U+Ip79+6N5s2bmxw3fvx4+Pn5YeLEiWatbRERUeUmeVg9+eSTsLW1Rdu2bTFixAh4eXnBxsamyPkGDBggaXlubm5F3vdydHSEu7s7AgICpGYREVEVIHlYjRkzxvj/WbNmFXsejUYjeVgRERFJJXlYbd++3ZodAICdO3da/TqIiEg8kodVhw4drNlBRERUIslfEUJERKSUUtes3n33XbMWptFosHDhwgoFERER/VOpwyo2NtashXFYERGRNZQ6rO7cuSNXBxERUYn4nhUREakehxUREakehxUREakehxUREakehxUREakehxUREakeh5UKua1KUzqBiEhVzPo2RZ1Oh82bNyMlJQV37tyBwWAwOV2j0eCrr76yaCAREZHkYXX48GGMHDkSd+/ehYuLS5HvogIeDysiIiJLkzyspk6dCnd3d+zcuROBgYHWbCIiIjIh+T2rixcvIiIigoOqGHyPiYjIuiQPq/r16yMvL8+aLURERMWSPKwmTpyIVatWcee2REQkO8nvWaWnp6NmzZpo0aIFBgwYAC8vL9jY2JicR6PR4O2337Z4JBERVW2Sh9XMmTON/1+1alWx5+GwIiIia5A8rE6fPm3NDiIiohJJHlY+Pj7W7CAiIiqRYrtbWrFiBdq1awdvb294e3uja9eu2LNnj1I5RESkYiWuWfXp0wfVqlXDli1bYGtri759+5a5MI1Gg23btkm64ieffBKzZs2Cn58f9Ho91q1bh+HDh+OHH37As88+K/0WEBFRpVfisDIYDNDr9cbDer2+zN0p/XNfgaXp3bu3yeEPPvgAK1euREJCAocVERGZKHFY7dy5s9TDlpSfn4/vvvsOubm5CA4Ottr1EBGRmDQ6nU766pCF/fbbb+jWrRvy8vLg5OSEFStWoHv37qVeJjk5WaY6aVrHOwIAEjrcs+gyLbk8IqLiqOG5xt/fX9L5FB1WDx48QGpqKrKysrBt2zasXr0aO3bsQEBAgFJJZivYL6DuVc9Sz5ecnCz5h+K2Kq3M5cnFnG41EbUbELed3fKyRLdSzzXlaTfr+6wszc7ODg0aNAAANG/eHImJiVi8eDG/E4uIiEyo6puC9Xo9Hjx4oHQGERGpjGJrVjNnzkS3bt3g6emJnJwcbNq0CfHx8YiLi1MqiYiIVEqxYZWeno7w8HBkZGTA1dUVTZo0waZNmxAWFqZUElmAmt5vI6LKQ7FhtWTJEqWumohUhH/gkBRmDasffvgBq1evRkpKCu7cuVPkQ8AajQZJSUkWDSQiIpI8rJYsWYJp06ahdu3aaNWqFRo3bmzNLiIiIiPJw2rRokVo3749Nm/eDDs7O2s2ERERmZC86XpmZiYGDhzIQUVERLKTPKyCgoJw9epVa7YQEREVS/KwmjNnDr799lscOXLEmj1ElULBbriIyDIkv2c1d+5cuLq6on///vDz84O3tzdsbGxMzqPRaPihXiIisjjJw+r8+fPQaDTw8vLC/fv3cfHixSLnKev7roiIiMpD8rA6e/asNTuIiIhKpKod2RIRERWHw4qIiFSvxJcB3d3dUa1aNVy/fh12dnZwd3cv8z0pjUaDzMxMi0fKgfsnIyJSrxKHVWRkJDQaDWxtbU0OExERya3EYRUVFVXqYSIiEovIryDxPSuqMvhBXSJxmf19VtevX8fp06eRlZUFvV5f5PRhw4ZZJIyIiKiA5GH14MEDvPnmm9i8eTP0ej00Go3x+6wKv5fFYUVERJYm+WXADz/8EJs3b0ZUVBR27NgBg8GAJUuWYOvWrejSpQsCAwPx448/WrPVLHzJh4io8pA8rDZv3owhQ4Zg0qRJxi9erFevHp577jls3LgRjo6OiI2NtVooERFVXZKHVUZGBtq0aQMAxs3Z8/LyADx+GbBfv37Ytm2bFRKJiKiqkzysatWqBZ1OBwBwcXFB9erVkZKSYjz94cOHyM3NtXggERGR5A0sAgMDkZCQAODxmlT79u2xePFiNG3aFHq9HsuXL0dgYKDVQomI5Cby55IqG8lrVq+88goMBoPxpb/o6Gjk5uaid+/e6NOnD+7du4c5c+ZIvuJPPvkEnTt3hre3N/z8/DBkyBCcO3fO/FtARESVnuQ1q549e6Jnz57Gw40aNUJiYiKOHj0KGxsbhISEwM3NTfIVx8fHY8yYMWjRogUMBgM+/PBD9O/fH8ePH4e7u7t5t4JUhX+NEpGlmf2h4MJcXV3Ru3dv4+H8/Pwi3x5cki1btpgcXrZsGXx8fHDs2DGToWhNfFIlIhKDRXa3dP/+faxYsQLNmzcv9zJycnKg1+vNWjsjIqKqocw1q/v372P37t24cuUK3N3d0aNHD2i1WgDAvXv3sGzZMixZsgQ3b95EgwYNyh0yZcoUBAYGIjg4uNTzJScnS1yio4TzFj6PlPMXvwypXZZtl495LdLvD2sr2iDn/Vqx67JUZ+t4RyR0uGeRZUlR3t8hpR8vJV+/8m2lMb/tn7dHudtXcL3+/v6Szl/qsLp+/Tp69+6NlJQU466VnJycsH79elSrVg1jx47FtWvXEBwcjI8//hh9+vQpV/TUqVNx7Ngx7N69u8yXEaXeMMSnlX3ewueRcv4SliGlKzk52bLtMjGrG5B8f1hbsd1y3q8VuC6z73MrdZir3N0KP95L7VbR7+I/lev+/uftUej2lae91GEVHR2Nq1ev4u2330a7du3w559/Yv78+XjnnXdw8+ZNBAQEYOXKlQgJCSl3dFRUFLZs2YLt27fD19e33MshIqLKq9Rh9cMPP2D48OGYOXOm8bg6derglVdeQbdu3bBu3TpUq1b+t70mT56MLVu2YMeOHWjYsGG5l0NERJVbqcMqIyMDrVq1MjmudevWAIARI0ZUaFBNmjQJGzZswDfffAM3Nzekp6cDePwyo7Ozc7mXS0RElU+p0yY/Px8ODg4mxxUcdnV1rdAVx8TE4O7du+jXrx+eeeYZ478vv/yyQsslUhr3+E9keWVuDZiSkoJffvnFeDg7OxvA4zfIilsDatmypaQrLtjPIBEVj58D5H1A/1XmsJo7dy7mzp1b5PjIyEiTwwaDARqNBrdv37ZcHVEVwSdlotKVOqwWLVokVwcREVGJSh1WL730klwdREREJbLI7paIiIisicOKiMjKuIVoxXFYERGR6nFYkaL4FycRScFhRarAoUVEpeGwIiIi1eOwIiIi1eOwUhBf+iIikobDiiot/jFAVHlUqWHFJy8iIjFVqWFFRERi4rCqxLgmSUSVBYeVBJX5Sb8y3zaqHPgYJYDDioiIBMBhRUSqwbUoKgmHVRWjpicDNbUQkbpxWFGlweFHVHlViWFlrScxPjkSEcmjSgwrS+KAIiKSn6LD6scff8TQoUPRuHFjuLm5Ye3atUrmEBGRSik6rHJzcxEQEICPPvoI1atXVzKFiCohvhJSeSg6rLp164bp06ejX79+qFbNOil8sFY9/JkTVT58z6qc+IRIRCQfDisrKm6glXRcZR1+arldaukgovKxVTrAXMnJyRLP6fj/53UsdDnHf1y+8OH//r91vCMSOtwz/t/0ek2XW3qX6fUVXM5tVdr/L7+4xqKXM9d/+4tbTmn3QfHMazHntpT2symPoj+bwh2mh62pPNdV/GPFcveJPMp3Xaa/h//9/Xh8uPzL/eeyTRX+PS/tMWqZ+886Pwfzl2n+77+1FFyvv7+/pPMLN6xKumFuq9Kge9Xzv0fEpz0+b3zafy9XcNw/z1PG/02u9x/LLbWr0HKSk5NL7inpOsqr0HKLLKfQcQVrG6Vdl7HbjOs2WWZpt6W0n015FPOzKfHnaE3lua4SHisWu09kYPZjpUAZv28mhyuy7BJOK7XbUvefFX4O5bq/S3sOlFF52vkyIBERqZ6iwyonJwdnzpzBmTNnoNfrkZqaijNnzuCvv/6SrYHvZVScSPehSK1VCX8uVBZFh9WpU6fQqVMndOrUCX///Tfmzp2LTp064cMPP1Qyi4jIiINUHRR9z6pjx47Q6XQWW16R962oylH6iUXp6ycqDxGeO/melUrxSY9IGiV/V0q77oLT1P67rPa+AhxWJDtRfjlI3fg4qloqxbBS84NWzW1ERKKoFMOK1EnKSySViSVvU2W8f4gqosoMq6r8y6+W266WDrIs/lyLUsMfaqXtxk3En1mVGVZEpC4iPmGaqyrcRrlwWBERkepxWJWD1L2pW2K5RGQ9/J0Th3DDig8uImkKf86HvzekJgXfZmEO4YYVyU+EJzq1N6q9r7Lj/S8+DisiIgmsPfCk7PGivA1qHNbmNnFYVSJqfECKgPeb+XifkdyEH1bW+qVR6peRTwJU2ajlMW3ue3dq6a7MzLmPhR9WaiPKzivVjvcfkfgsuXGP0MOqMjyhWeo2lLUcc/+iFP2+Lc/WRkRyqcjvl1zvW6ntOUDoYSU3Nfzw+HkuqghRf/aidpPlcFjJwJq/aFXtl7iq3d6y8P4gpcn1GFT0m4KtQYRvvKxqLPkSJZGltY53hM7f9Dg5HpOFr6Miz1mWes4rrqfgODU8p3LNSiZ8D4WISiLnH2yi/nHIYWVBatwSsDJ9RQCJQ+ratBwbGPGxbjlK3peVclipcWhUNRySVZPcL59VNpXp98bSzZVyWFmKiA8QImvj7wUpQfFhFRMTg6ZNm0Kr1SI0NBQ//fST0knF4i+oMiq6cUZl+MxYgcpyO6SoSre1Mij887LWVygpOqy2bNmCKVOm4N1338WRI0cQHByMF198EX/99ZfVrpO/BFQcazwulNgVmMiPb0vtCkmtf8BUdHdPlv4wsGiPFUWH1aJFi/DSSy9h1KhReOaZZ7BgwQJotVrExsYqmVUpVLZ9oKltDemfPXLtkbus49+dz40AABe2SURBVJSiphZShpTHQEUeJ4oNqwcPHiApKQldunQxOb5Lly44fvx4mZe35C+HXL9o5j65lbahiKWbi7sut1Vpxk3ulfjrjE+AVFhVfayVp02u+0rO+02j0+kMsl1bIdevX0fjxo2xc+dOtG/f3nj8vHnzsHHjRpw8eVKJLCIiUiHFN7DQaDQmhw0GQ5HjiIioalNsWNWqVQs2NjbIyMgwOf7WrVvw8PBQqIqIiNRIsWFlZ2eHoKAgHDp0yOT4Q4cOoU2bNgpVERGRGim6I9vx48fj9ddfR8uWLdGmTRvExsbixo0bePXVV5XMIiIilVF0WA0cOBC3b9/GggULkJ6ejsaNGyMuLg4+Pj5KZhERkcootjUgERGRVKr8PqukpCQEBQUpnVFuubm5SE5ORuPGjWFvb4+///4bO3bsgF6vR6dOnVCvXj2lE0uUm5uLpKQkpKenw8bGBvXr10ezZs1Uv4XmpUuXcPz4cWRkZECj0cDDwwNt2rSBn5+f0mnlVvCzKPzRDhHo9XqkpaXB29tb6ZRiPXjwAHZ2dsbDx44dw/3799G2bVuT49UuPDwcs2bNUvXzSXF0Oh0uX74MrVYLT0/p35OlyjUrd3d3+Pr6YtSoUXjppZdQp04dpZMkS0xMxMCBA5GVlQUfHx9s3boVw4YNQ2pqKjQaDWxsbLB582a0atVK6VQTer0eM2fORExMDPLy8gA8/hgBAHh5eWH+/Pno2bOnkonFysrKwrhx47B79244OTmhdu3aMBgMyMzMxL1799CjRw8sXboUrq6uSqea7ezZswgNDcXt27eVTjGRl5eHqKgobNu2DW5ubnjttdcQERFhPD0jIwONGjVSXff169cxcuRIJCYmonXr1li/fj3Gjh2LAwcOAAB8fX3xn//8R3VP/klJScUe37VrV8TExKB+/foAoMo/8GfPno1JkybB0dERDx8+xKRJk7BmzRrjR5R69eqFmJgYODg4lLksxT9nVZLg4GB8+umnePbZZzFy5EjjA0rtZs2ahW7duiEpKQkvvPACBg0ahMaNGyMlJQUpKSno3r07Zs+erXRmEbNnz8aePXsQGxuLLVu2ICQkBDNnzsTx48cxdOhQvPLKKzh48KDSmUVERkYiJSUFu3btQmpqKpKSknD69GmkpqZi165dSElJQWRkpNKZlcr8+fOxZ88eTJ06FSNGjMDChQsRHh4OvV5vPE/BHzpqMmPGDNjY2GDt2rXw9PTE0KFDkZubi99++w1nzpyBVqvFJ598onRmEZ07d0aXLl3QuXNnk3+PHj3CK6+8YjxdjT777DPk5uYCAL744gvs2LEDsbGxOHPmDNasWYPExER88cUXkpal2jWrP/74A87OztiyZQu+/vprnDhxAl5eXhg5ciSGDx9u1uqjnOrXr4/9+/fD398f9+/fx5NPPom9e/eiZcuWAIDff/8dvXr1wpUrVxQuNdW4cWOsXLkS7dq1AwBcu3YNwcHBuHTpEuzt7TF//nzs378fe/fuVbjUlI+PD7Zs2VLimuqJEycwaNAgXL16VeaystWsWVPS+dS2hhIUFIQFCxaga9euAIC//voLgwYNQpMmTRATE4Nbt26pcs2qUaNGWLNmDVq3bo07d+6gQYMG+O677xAaGgoAOHz4MN55550S12SU0r59e3h5eeHf//437O3tATz+Y6Bly5bYtGkTGjRoAACq3DCt4Lncw8MDHTt2RHh4OEaOHGk8fevWrfjoo48k7WJPtWtWAFC9enUMHz4ce/bswU8//YRevXphyZIlaNasGYYMGaJ0XrEKv7dT8H8bGxvjcTY2Nqr8qzMnJwdPPvmk8bBWq0VeXh50Oh0A4Pnnn8evv/6qVF65Vaum3od49erVMXHiRKxatarYf7NmzVI6sVjp6elo2LCh8bC3tze2b9+Oc+fOYfTo0Xj48KGCdSXT6XTGl/jc3d3h6Oho8r5agwYNcOPGDaXySnTw4EH4+Phg1KhRuHv3Lnx8fIwv/dWtWxc+Pj6qHFQFCp4H09LSjH+0F2jRooXkb9lQ5W9ycW/mN27cGPPmzcP58+exaNEi5OTkKFBWtqCgIHz66ae4evUqFixYAF9fXyxfvtx4+rJly9C4cWMFC4sXEBCAuLg44+FNmzbByckJWq0WwOP3tNT45nOPHj3w9ttvIyEhochpCQkJeOedd1T5XhsABAYGwt3dHf369Sv233PPPad0YrG0Wm2RVwbq1KmD77//HufOncO4ceMUKitd7dq1kZ6ebjw8duxYuLu7Gw9nZWXByclJibRS2dvbY8GCBZg2bRoGDRqExYsXK51klpUrV+Krr76CnZ1dkbXt7Oxsyc8rqtwasLQ1D3t7ewwZMkS1a1bTp0/HoEGDsH79etSuXRvbt2/Hm2++CX9/f2g0Gty9exfr169XOrOIqVOnYvDgwdi5cyccHBxw8uRJREdHG08/cOAAmjZtqmBh8ebPn4/XXnsN3bp1g4uLC2rVqgWNRoNbt24hJycHYWFhmDdvntKZxeratSuys7NLPN3d3R1Dhw6VsUiajh07YuPGjUWGqVarxbZt29C7d29lwsoQGBiIhIQE41/3M2fONDn92LFjCAgIUKBMmt69eyMoKAjh4eHYt2+f0jmSeHl5Ye3atQAe77XozJkz6NChg/H0o0ePwt/fX9KyVPme1bfffosXXnjB+PqsaAo2XX/66afh7OyMvLw8xMXFIS8vD507d5b8w5Hbr7/+iq1bt+L+/fsICwtD586dlU6S7I8//sCJEyeM+5qsU6cOgoODTV6uIsu4evUqkpOTERYWVuzpN27cwMGDB/HSSy/JXFa6gj+CS/oYRkJCAhwcHBAYGChnltn0ej0WLlyII0eOYPHixap+CbAsCQkJsLOzQ7Nmzco8ryqHFRERUWGqfBkQePxX0A8//FDkg54hISEIDQ1V9YdURW0Xtbs0Op0Ou3btwrBhw5ROMZuo7eyWl6jdgHntqlyzunbtGoYMGYLffvsNzzzzDDw8PGAwGHDr1i1cuHABgYGBWLduncnWa2oharuo3WVR6wdrpRC1nd3yErUbMK9dlWtW7777LmrUqIGzZ88W+TxVWloaxo0bh0mTJuHbb79VqLBkoraL2l3WZq9q3BS5gKjt7JaXqN2AZdtVuWbl6emJXbt2lbj12enTp9GrVy+kpaXJXFY2UdtF7XZ3dy/15cmC3bqo8a9OUdvZLS9RuwHLtqtyzcrBwQF37twp8XSdTidpX1JKELVd1G5XV1dERUUhJCSk2NOTk5Px+uuvy1wljajt7JaXqN2AZdtVOawGDhyIiIgIREdHo3Pnzsbd0ty+fRuHDh0yfpZJjURtF7W7adOmyMvLK3EnnmrdYwggbju75SVqN2DZdlUOqzlz5iA/Px8RERF49OiRcXdF+fn5sLW1xciRI00+sKomoraL2j1o0CDcu3evxNO1Wi0mT54sY5F0orazW16idgOWbVfle1YFsrOzkZSUZPJBz6CgICG+7kHUdlG7iahyU/WwKiwtLQ316tVT9Y5JSyJqO7vlJ2o7u+UlajdQ/nZhbmlISIgqv+ZBClHb2S0/UdvZLS9Ru4HytwszrNT6BqIUorazW36itrNbXqJ2A+VvF2ZYERFR1WUzZcqUmUpHSNW2bVtVftZHClHb2S0/UdvZLS9Ru4HytQuzgQUREVVdqnwZsOD7WoYOHYpVq1YBAL755hu0bNkSzZs3x/Tp0/HgwQOFK4snaju75SdqO7vlJWo3YNl2Vb4MOGfOHCxduhSBgYFYv3497ty5g48//hijR49GUFAQYmNjkZubi06dOimdWoSo7eyWn6jt7JaXqN2AZdtV+TJgs2bNMG/ePPTo0QPnz59Hu3btsHTpUgwePBgAsH37dkyfPh2nTp1SuLQoUdvZLT9R29ktL1G7Acu2q/JlwPT0dDRp0gQA0KhRI9jY2Jh81XSzZs2Qnp6uVF6pRG1nt/xEbWe3vETtBizbrsphpdVq8fvvvwN4vFfe/Px8XLhwwXj6+fPnUbt2baXySiVqO7vlJ2o7u+Ulajdg2XZV7sj2xRdfxLhx49CjRw8cPXoUEydOxPvvv4+MjAxUq1YNn332GZ5//nmlM4slaju75SdqO7vlJWo3YNl2VW5g0aFDB+Tn5+PixYt4/vnnERkZCa1Wi3nz5uHgwYMICwtDdHQ07OzslE4tQtR2dstP1HZ2y0vUbsCy7arcwIKIiKgwVb4MWFh+fj4yMzOh0WhQs2ZN4/csiUDUdnbLT9R2dstL1G6g4u2q3MACeLxJY/fu3VGvXj00atQIzzzzDOrVq4fu3btjx44dSueVStR2dstP1HZ2y0vUbsBy7ap8GXDVqlWIjIzEsGHDEBYWBg8PDxgMBty6dQsHDx7E+vXrMX/+fIwaNUrp1CJEbWe3/ERtZ7e8RO0GLNuuymHVvHlz/M///A9efvnlYk//+uuv8cknnyApKUnmsrKJ2s5u+Ynazm55idoNWLZdlS8DXr9+HW3bti3x9JCQENy4cUPGIulEbWe3/ERtZ7e8RO0GLNuuymHVqFEjrFy5ssTTV61ahUaNGslYJJ2o7eyWn6jt7JaXqN2AZdtV+TJgfHw8hgwZgnr16qFz586oU6cONBoN0tPTcfjwYVy7dg1xcXFo166d0qlFiNrObvmJ2s5ueYnaDVi2XZXDCgD+/PNPxMbGIiEhARkZGQCAOnXqIDg4GK+++irq16+vcGHJRG1nt/xEbWe3vETtBizXrtphRUREVECV71kV591330VmZqbSGeUiaju75SdqO7vlJWo3UP52YYZVXFwc7t69q3RGuYjazm75idrObnmJ2g2Uv12YYWUwiPtqpajt7JafqO3slpeo3UD524UZVkREVHVxAwsiIlI91e91HQDy8vKwe/du/PXXX/Dx8UH37t3h4OCgdJYkorazW36itrNbXqJ2AxVrV+WwioiIQK9evdC3b1+kpKSgb9++uHXrFurWrYv09HR4eHjg+++/h6+vr9KpRYjazm75idrObnmJ2g1Ytl2V71nt3bsXTz/9NADg/fffR0BAAC5cuIBTp07hjz/+QFBQEKKiohSuLJ6o7eyWn6jt7JaXqN2AZdtVuWaVm5uL6tWrAwASExOxdu1auLq6AgCcnZ0RFRWFnj17KplYIlHb2S0/UdvZLS9RuwHLtqtyzcrf3x8nT54EALi6ukKn05mcnpWVBY1Go0RamURtZ7f8RG1nt7xE7QYs224zZcqUmZYOrCgnJyfMmDEDgYGBaNq0KaKjo9GwYUPY29vj1KlTeO+999C5c2dV/jUhaju75SdqO7vlJWo3YNl21W66vnTpUkRHR0Ov1yM/Px+PHj0yntazZ08sX74cTk5OChaWTNR2dstP1HZ2y0vUbsBy7aodVsDjVcRDhw4hJSUFer0eWq0WISEh8PPzUzqtTKK2s1t+orazW16idgOWaVf1sCIiIgJUuoFFWXJzc/Hjjz8qnVEuorazW36itrNbXqJ2A+a1CzmsLl++jL59+yqdUS6itrNbfqK2s1teonYD5rULOayIiKhqUeWHgmvWrKl0QrmJ2s5u+Ynazm55idoNWLZdlcOqevXqiIiIQGBgYLGnX716FTNmzJC5ShpR29ktP1Hb2S0vUbsBy7arclgFBgbC3d0d/fr1K/b0s2fPylwknajt7JafqO3slpeo3YBl21X5nlXXrl2RnZ1d4unu7u4YOnSojEXSidrObvmJ2s5ueYnaDVi2nZ+zIiIi1VPlmhUREVFhqnzPCnj8YbFNmzbh+PHjyMjIgEajgYeHB0JCQvDCCy+odj9YgLjt7JafqO3slpeo3YDl2lX5MuD58+cxYMAA5OTkoF27dvDw8IDBYMCtW7fw888/w9nZGVu2bEGjRo2UTi1C1HZ2y0/UdnbLS9RuwLLtqhxWffr0gYeHB5YsWQIHBweT0/Ly8vDGG28gIyMDO3bsUKiwZKK2s1t+orazW16idgOWbVflsKpXrx4OHTpU4rQ9d+4cwsLCcP36dZnLyiZqO7vlJ2o7u+Ulajdg2XZVbmDh5uaGixcvlnj6pUuX4ObmJmORdKK2s1t+orazW16idgOWbVflNwXfvXsXs2bNAvD4E9D5+fnIycnBpUuXsG7dOnzwwQcIDw9Hhw4dFC4tStR2dstP1HZ2y0vUbsCy7ap8GRAAPvvsMyxduhTp6enQaDQAAIPBAK1Wi4iICLzzzjsKF5ZM1HZ2y0/UdnbLS9RuwHLtqh1WBVJSUpCRkQEAqFOnDnx9fZUNMoOo7eyWn6jt7JaXqN1AxdtVP6yIiIhUuYEFAOh0OuzZswfHjx+HwWA6T3NzczFv3jyFysomaju75SdqO7vlJWo3YLl2Va5Z/f777+jfvz9u3boFvV6PZs2a4euvv4aPjw8AICMjA40aNcLt27cVLi1K1HZ2y0/UdnbLS9RuwLLtqlyzmjVrFlq3bo2rV6/i999/h6+vL3r06IFLly4pnVYmUdvZLT9R29ktL1G7Acu2q3JYnTx5EtOmTYOTkxPq1q2L//3f/0X//v3Rp0+fUrfZVwNR29ktP1Hb2S0vUbsBy7arcke2Dx48MG7iWODDDz+EwWBA7969ERMTo1BZ2URtZ7f8RG1nt7xE7QYs267KYfX000/j1KlTRXbRMXfuXOj1egwfPlyhsrKJ2s5u+Ynazm55idoNWLZdlS8D9unTB5s3by72tHnz5mHw4MFFtipRC1Hb2S0/UdvZLS9RuwHLtqtya0AiIqLCVLlmRUREVBiHFRERqR6HFRERqR6HFZGVrF27Fm5ubsZ/Wq0WjRo1wsCBA7F06VLcvXtX6UQiYahy03WiymTKlCl46qmn8PDhQ2RkZCA+Ph5RUVFYtGgR1q1bh2effVbpRCLV47AisrKwsDC0bt3aeHjixIk4fPgwhg4dimHDhuHEiROoXr26goVE6seXAYkUEBoaivfeew9//fUX4uLiAAC//vorIiIiEBQUBK1WCz8/P4wZMwapqanGyxV8DfiiRYuKLPP8+fNwc3PD8uXLZbsdRHLhsCJSyJAhQwAABw8eBAAcOnQIycnJGDx4MObPn48RI0Zg37596Nu3L/7++28AgJ+fH9q0aYMNGzYUWd6GDRvwxBNP4IUXXpDvRhDJhC8DEinE09MTrq6uuHLlCgBgzJgxeOutt0zO06NHD/Ts2RPbt2/H4MGDAQDDhg3DhAkTcP78eeNubAwGAzZu3IiwsDDUqlVL3htCJAOuWREpyNnZGTk5OQAAR0dH4/E5OTm4ffs2GjZsiBo1aiApKcl42oABA+Dg4GCydhUfH4/U1FQMHTpUvngiGXFYESkoJycHzs7OAB5/o+qECRPw1FNPwcvLCw0aNICfnx+ysrKQlZVlvEyNGjXQq1cvbNy40bhftbi4OLi6uqJHjx6K3A4ia+PLgEQKSUtLQ3Z2Nho0aAAAGD16NH766Se8+eabaNq0KVxcXKDRaDB69Gjo9XqTyw4bNgxbtmzBjz/+iNatW+P7779H//794eDgoMRNIbI6DisihRS8jNelSxfodDocPHgQU6ZMwZQpU4znycvLg06nK3LZLl26oG7dutiwYQNu3bqF7Oxs4wYbRJURhxWRAg4fPowFCxagfv36GDx4MB48eAAARb4uYfHixUXWqgDAxsYGL774IlavXo1r167Bx8cH7dq1k6WdSAkcVkRWduDAAVy+fBmPHj3CzZs3ceTIERw6dAje3t5Yt24dHBwc4ODggA4dOuCLL77Aw4cP4e3tjZ9//hk//fQTatasWexyhw0bhi+//BIHDhzApEmTinwjK1FlwmFFZGUfffQRAMDOzg7u7u4ICAjA3LlzMXz4cLi4uBjPFxMTgylTpmDVqlV49OgR2rVrh23btqFfv37FLjcgIADNmjXD6dOn+RIgVXr88kUigXXt2hV6vR4HDhxQOoXIqrjpOpGgzp07h4SEBAwbNkzpFCKr45oVkWDOnTuHpKQkLF26FGlpaTh9+rTxs1pElRXXrIgE8/3332P8+PG4d+8eVq5cyUFFVQLXrIiISPW4ZkVERKrHYUVERKrHYUVERKrHYUVERKrHYUVERKrHYUVERKr3f0sZG2Ko6+yrAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "weather_df.plot.bar(width=1)\n",
    "\n",
    "\n",
    "plt.xticks(np.arange(1,365,45),rotation=90)\n",
    "plt.xlabel(\"Day\")\n",
    "plt.ylabel(\"Rain in Inches\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Precipitation</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>344.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>0.460494</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>0.713201</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>0.010000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>0.215000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>0.657500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>6.700000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       Precipitation\n",
       "count     344.000000\n",
       "mean        0.460494\n",
       "std         0.713201\n",
       "min         0.000000\n",
       "25%         0.010000\n",
       "50%         0.215000\n",
       "75%         0.657500\n",
       "max         6.700000"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "weather_df.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "9"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "station_count = session.query(Station.station).count()\n",
    "station_count"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[('USC00519281', 2772),\n",
       " ('USC00519397', 2724),\n",
       " ('USC00513117', 2709),\n",
       " ('USC00519523', 2669),\n",
       " ('USC00516128', 2612),\n",
       " ('USC00514830', 2202),\n",
       " ('USC00511918', 1979),\n",
       " ('USC00517948', 1372),\n",
       " ('USC00518838', 511)]"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "station_groupby= session.query(Measurment.station, func.count(Measurment.date)).group_by(Measurment.station).order_by(func.count(Measurment.date).desc()).all()\n",
    "station_groupby"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Station USC00519281 Lowest temp is [(54.0,)], Highest temp is [(85.0,)] and average temp is [(71.66378066378067,)]\n"
     ]
    }
   ],
   "source": [
    "highest = session.query(func.max(Measurment.tobs)).filter(Measurment.station==\"USC00519281\").all()\n",
    "lowest = session.query(func.min(Measurment.tobs)).filter(Measurment.station==\"USC00519281\").all()\n",
    "avg = session.query(func.avg(Measurment.tobs)).filter(Measurment.station==\"USC00519281\").all()\n",
    "print(f'Station USC00519281 Lowest temp is {lowest}, Highest temp is {highest} and average temp is {avg}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "('2017-08-18')"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "session.query(Measurment.date).filter(Measurment.station==\"USC00519281\").order_by(Measurment.date.desc()).first()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Temp</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Date</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2016-08-19</th>\n",
       "      <td>79.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2016-08-20</th>\n",
       "      <td>81.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2016-08-21</th>\n",
       "      <td>79.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2016-08-22</th>\n",
       "      <td>78.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2016-08-23</th>\n",
       "      <td>77.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2017-08-14</th>\n",
       "      <td>77.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2017-08-15</th>\n",
       "      <td>77.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2017-08-16</th>\n",
       "      <td>76.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2017-08-17</th>\n",
       "      <td>76.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2017-08-18</th>\n",
       "      <td>79.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>356 rows × 1 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "            Temp\n",
       "Date            \n",
       "2016-08-19  79.0\n",
       "2016-08-20  81.0\n",
       "2016-08-21  79.0\n",
       "2016-08-22  78.0\n",
       "2016-08-23  77.0\n",
       "...          ...\n",
       "2017-08-14  77.0\n",
       "2017-08-15  77.0\n",
       "2017-08-16  76.0\n",
       "2017-08-17  76.0\n",
       "2017-08-18  79.0\n",
       "\n",
       "[356 rows x 1 columns]"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "active_station = session.query(Measurment.date, Measurment.tobs).filter(Measurment.station==\"USC00519281\").filter(Measurment.date > '2016-08-18').all()\n",
    "\n",
    "station_USC00519281_df = pd.DataFrame(active_station, columns=['Date', 'Temp'])\n",
    "station_USC00519281_df.set_index('Date', inplace=True, )\n",
    "station_USC00519281_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0.5, 0, 'Temp(f)')"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "station_USC00519281_df.plot.hist(bins=12)\n",
    "plt.xlabel(\"Temp(f)\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
